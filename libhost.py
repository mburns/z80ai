"""
Minimal host-environment shims for running generated binaries under
:mod:`libz80emu`.

Each host installs hooks at the entry points the corresponding build script
targets and turns console traffic into plain Python strings, so a test can do::

    out = run_cpm(open('CHAT.COM','rb').read(), stdin=['hello', '!'])

Only the handful of entry points the builders actually call are implemented;
anything else raises, so a typo in a build script fails loudly instead of
silently reading zeroes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, ClassVar

# Entry points and memory maps come from the target modules rather than being
# restated: the emulator's idea of where BDOS sits and the code generator's have
# to agree, and the surest way to guarantee that is to have only one of them.
from libagon import AGON_CR
from libagon import MOS_API as AGON_MOS_API
from libagon import MOS_GETKEY as AGON_MOS_GETKEY
from libagon import MOS_LOAD as AGON_MOS_LOAD
from libagon import MOS_OUTCHAR as AGON_MOS_OUTCHAR
from libcpc import CPC_CR, CPC_HIMEM
from libcpc import KM_WAIT_CHAR as CPC_KM_WAIT_CHAR
from libcpc import ORG_ADDR as CPC_ORG_ADDR
from libcpc import SCR_SET_MODE as CPC_SCR_SET_MODE
from libcpc import TXT_OUTPUT as CPC_TXT_OUTPUT
from libcpm import BDOS, CPM_CMDLINE, TPA, TPA_TOP
from libnext import NEXT_REG_CPU_SPEED, NEXT_REG_SELECT, NEXT_REG_VALUE
from libnext import SPEEDS as NEXT_SPEEDS
from libz80emu import Z80, Z80Error
from libzx import ORG_ADDR as ZX_DEFAULT_ORG
from libzx import (
    ZX_CHAN_OPEN,
    ZX_CLS,
    ZX_ENTER,
    ZX_KEY_INPUT,
    ZX_PRINT_A,
    ZX_RAM_TOP,
)

#: Where a transient program's stack starts, a little below the BDOS. Real CP/M
#: hands over whatever the CCP was using; anywhere clear of the image will do.
CPM_STACK = 0xE000

# --- CP/M --------------------------------------------------------------------


#: The key that ends a line, on every machine here. Asserted rather than
#: written as 13, because the shared shim below has to agree with all three
#: targets and there is no reason to trust that it does.
ENTER = ZX_ENTER
assert ENTER == CPC_CR == AGON_CR


class _StdinKeys:
    """Console traffic for a chat-driven host: queued input, collected output.

    ZXHost, CPCHost and AgonHost each had their own copy of both halves. The
    key feed was byte-identical between ZX and CPC and differed from Agon only
    in who pops the return address; the character collector was identical in
    all three.
    """

    stdin: list[str]
    output: list[str]
    finished: bool

    def _feed_key(self, cpu: Z80) -> bool:
        """Put the next queued character in A. False when the input ran out.

        A caller that returns to the guest itself - the RST-based hosts - pops
        only when this returned True; a halted CPU has no return address to
        take.
        """
        if not self.stdin:
            self.finished = True
            cpu.halted = True
            return False
        line = self.stdin[0]
        if line == "":
            self.stdin.pop(0)
            cpu.a = ENTER
        else:
            self.stdin[0] = line[1:]
            cpu.a = ord(line[0])
        return True

    def _emit_char(self, cpu: Z80) -> bool:
        """Collect the character in A and return to the guest."""
        self.output.append(chr(cpu.a))
        cpu.pc = cpu._pop()
        return True


class CPMExit(Exception):
    """Raised internally when the program returns to CCP."""


class CPMHost:
    """A CP/M 2.2 BDOS subset: functions 0, 1, 2, 6, 9, 10 and 11."""

    def __init__(self, cmdline: str = "", stdin: list[str] | None = None) -> None:
        self.cpu = Z80()
        self.output: list[str] = []
        self.stdin = list(stdin or [])
        self.finished = False
        self._set_cmdline(cmdline)

        # 0000h: warm boot. 0005h: BDOS entry, a JP to the real BDOS at the top
        # of the TPA - programs read the address there to size the TPA.
        self.cpu.poke(0x0000, 0xC3)  # JP 0000h - the hook intercepts first
        self.cpu.poke(BDOS, 0xC3)
        self.cpu.poke(BDOS + 1, TPA_TOP & 0xFF)
        self.cpu.poke(BDOS + 2, TPA_TOP >> 8)
        self.cpu.hooks[0x0000] = self._warm_boot
        self.cpu.hooks[BDOS] = self._bdos
        self.cpu.sp = CPM_STACK

    def _set_cmdline(self, cmdline: str) -> None:
        text = cmdline.upper()[:126]
        self.cpu.poke(CPM_CMDLINE, len(text))
        for i, ch in enumerate(text):
            self.cpu.poke(CPM_CMDLINE + 1 + i, ord(ch))
        self.cpu.poke(CPM_CMDLINE + 1 + len(text), 0)

    def _warm_boot(self, cpu: Z80) -> bool:
        self.finished = True
        cpu.halted = True
        return True

    def _putc(self, code: int) -> None:
        self.output.append(chr(code))

    def _bdos(self, cpu: Z80) -> bool:
        fn = cpu.c
        if fn == 0:  # P_TERMCPM
            self.finished = True
            cpu.halted = True
            return True
        if fn == 1:  # C_READ - read char with echo
            ch = self._next_char()
            cpu.a = cpu.l = ch
            self._putc(ch)
        elif fn == 2:  # C_WRITE
            self._putc(cpu.e)
        elif fn == 6:  # C_RAWIO
            if cpu.e == 0xFF:
                cpu.a = cpu.l = self._next_char(nonblocking=True)
            else:
                self._putc(cpu.e)
        elif fn == 9:  # C_WRITESTR
            addr = cpu.de
            for _ in range(255):
                ch = cpu.peek(addr)
                if ch == ord("$"):
                    break
                self._putc(ch)
                addr += 1
        elif fn == 10:  # C_READSTR
            self._read_line(cpu, cpu.de)
        elif fn == 11:  # C_STAT
            cpu.a = cpu.l = 0xFF if self.stdin else 0x00
        else:
            raise Z80Error(f"unimplemented BDOS function {fn}")
        cpu.pc = cpu._pop()
        return True

    def _next_char(self, nonblocking: bool = False) -> int:
        if not self.stdin:
            if nonblocking:
                return 0
            self.finished = True
            raise CPMExit()
        line = self.stdin[0]
        if line == "":
            self.stdin.pop(0)
            return 13
        self.stdin[0] = line[1:]
        return ord(line[0])

    def _read_line(self, cpu: Z80, buf: int) -> None:
        maxlen = cpu.peek(buf)
        if not self.stdin:
            self.finished = True
            raise CPMExit()
        line = self.stdin.pop(0)[:maxlen]
        cpu.poke(buf + 1, len(line))
        for i, ch in enumerate(line):
            cpu.poke(buf + 2 + i, ord(ch))
        self.output.append(line)

    def run(self, image: bytes, max_cycles: int = 2_000_000_000) -> str:
        self.cpu.load(TPA, image)
        self.cpu.pc = TPA
        # CPMExit unwinds out of the BDOS hook when input runs out.
        with contextlib.suppress(CPMExit):
            self.cpu.run(max_cycles=max_cycles)
        return "".join(self.output)


def run_cpm(
    image: bytes,
    cmdline: str = "",
    stdin: list[str] | None = None,
    max_cycles: int = 2_000_000_000,
) -> tuple[str, CPMHost]:
    """Run a .COM image; returns (console output, host) for cycle inspection."""
    host = CPMHost(cmdline=cmdline, stdin=stdin)
    return host.run(image, max_cycles=max_cycles), host


# --- ZX Spectrum -------------------------------------------------------------


class ZXHost(_StdinKeys):
    """Stubs for the handful of 48K ROM routines the TAP build calls."""

    def __init__(self, stdin: list[str] | None = None, org: int = ZX_DEFAULT_ORG) -> None:
        self.cpu = Z80()
        self.output: list[str] = []
        self.stdin = list(stdin or [])
        self.org = org
        self.finished = False
        # BASIC's stack lives below the code once RAMTOP is moved down with
        # CLEAR, so put ours there too rather than somewhere the image covers.
        self._stack = org - 0x20
        self._exit = org - 0x40
        for addr, fn in (
            (ZX_PRINT_A, self._print_a),
            (ZX_CLS, self._ret),
            (ZX_CHAN_OPEN, self._ret),
            (ZX_KEY_INPUT, self._key_input),
        ):
            self.cpu.hooks[addr] = fn
        self.cpu.sp = self._stack

    @staticmethod
    def _ret(cpu: Z80) -> bool:
        cpu.pc = cpu._pop()
        return True

    def _print_a(self, cpu: Z80) -> bool:
        return self._emit_char(cpu)

    def _key_input(self, cpu: Z80) -> bool:
        if not self._feed_key(cpu):
            return True
        cpu.pc = cpu._pop()
        return True

    def run(self, image: bytes, max_cycles: int = 2_000_000_000) -> str:
        if self.org + len(image) > ZX_RAM_TOP:
            raise Z80Error(
                f"image of {len(image):,} bytes does not fit in RAM at {self.org:#06x}"
            )
        self.cpu.load(self.org, image)
        self.cpu.pc = self.org
        # RANDOMIZE USR returns to BASIC; here a RET lands on a HALT.
        self.cpu.poke(self._exit, 0x76)
        self.cpu.sp = self._stack
        self.cpu.poke(self._stack, self._exit & 0xFF)
        self.cpu.poke(self._stack + 1, (self._exit >> 8) & 0xFF)
        self.cpu.run(max_cycles=max_cycles)
        return "".join(self.output)


def run_zx(
    image: bytes,
    stdin: list[str] | None = None,
    org: int = ZX_DEFAULT_ORG,
    max_cycles: int = 2_000_000_000,
) -> tuple[str, ZXHost]:
    host = ZXHost(stdin=stdin, org=org)
    return host.run(image, max_cycles=max_cycles), host


# --- ZX Spectrum Next --------------------------------------------------------


class NextHost(ZXHost):
    """A Spectrum whose Next registers are watched rather than ignored.

    The Next is Spectrum-compatible, so the ROM stubs are inherited whole. What
    is added is the select/value port pair: recording the writes is what lets a
    test assert the build really asks for 28MHz, which is otherwise invisible -
    the emulator has one clock and does not speed up.
    """

    def __init__(self, stdin: list[str] | None = None,
                 org: int = ZX_DEFAULT_ORG) -> None:
        super().__init__(stdin=stdin, org=org)
        #: Next register number -> the last value written to it.
        self.registers: dict[int, int] = {}
        self._selected: int | None = None
        self.cpu.io_write = self._io_write

    def _io_write(self, port: int, value: int) -> None:
        if port == NEXT_REG_SELECT:
            self._selected = value
        elif port == NEXT_REG_VALUE and self._selected is not None:
            self.registers[self._selected] = value

    @property
    def cpu_speed(self) -> str | None:
        """The clock the program asked for, as a key of ``libnext.SPEEDS``."""
        value = self.registers.get(NEXT_REG_CPU_SPEED)
        if value is None:
            return None
        return next((k for k, v in NEXT_SPEEDS.items() if v == value), None)


def run_next(
    image: bytes,
    stdin: list[str] | None = None,
    org: int = ZX_DEFAULT_ORG,
    max_cycles: int = 2_000_000_000,
) -> tuple[str, NextHost]:
    host = NextHost(stdin=stdin, org=org)
    return host.run(image, max_cycles=max_cycles), host


# --- Amstrad CPC -------------------------------------------------------------


class CPCHost(_StdinKeys):
    """Stubs for the firmware jumpblock entries the CPC build calls.

    The jumpblock lives in RAM on a real CPC, so a program reaches the firmware
    by CALLing a fixed address rather than through a ROM.  That makes the shim
    the same shape as the Spectrum's: hook the addresses, answer, RET.
    """

    def __init__(self, stdin: list[str] | None = None,
                 org: int = CPC_ORG_ADDR) -> None:
        self.cpu = Z80()
        self.output: list[str] = []
        self.stdin = list(stdin or [])
        self.org = org
        self.finished = False
        self.mode: int | None = None
        # A CPC program is entered below the firmware workspace, so unlike the
        # Spectrum there is no room under the image: park the stack and the
        # exit trampoline just below HIMEM instead.
        self._stack = CPC_HIMEM - 0x20
        self._exit = CPC_HIMEM - 0x40
        for addr, fn in (
            (CPC_TXT_OUTPUT, self._txt_output),
            (CPC_KM_WAIT_CHAR, self._km_wait_char),
            (CPC_SCR_SET_MODE, self._scr_set_mode),
        ):
            self.cpu.hooks[addr] = fn
        self.cpu.sp = self._stack

    def _txt_output(self, cpu: Z80) -> bool:
        return self._emit_char(cpu)

    def _scr_set_mode(self, cpu: Z80) -> bool:
        self.mode = cpu.a
        cpu.pc = cpu._pop()
        return True

    def _km_wait_char(self, cpu: Z80) -> bool:
        if not self._feed_key(cpu):
            return True
        cpu.pc = cpu._pop()
        return True

    def run(self, image: bytes, max_cycles: int = 2_000_000_000) -> str:
        if self.org + len(image) > CPC_HIMEM:
            raise Z80Error(
                f"image of {len(image):,} bytes does not fit below HIMEM "
                f"at {self.org:#06x}"
            )
        self.cpu.load(self.org, image)
        self.cpu.pc = self.org
        # RUN" calls the program; returning drops onto a HALT.
        self.cpu.poke(self._exit, 0x76)
        self.cpu.sp = self._stack
        self.cpu.poke(self._stack, self._exit & 0xFF)
        self.cpu.poke(self._stack + 1, (self._exit >> 8) & 0xFF)
        self.cpu.run(max_cycles=max_cycles)
        return "".join(self.output)


def run_cpc(
    image: bytes,
    stdin: list[str] | None = None,
    org: int = CPC_ORG_ADDR,
    max_cycles: int = 2_000_000_000,
) -> tuple[str, CPCHost]:
    host = CPCHost(stdin=stdin, org=org)
    return host.run(image, max_cycles=max_cycles), host


# --- Agon / eZ80 MOS ---------------------------------------------------------
#
# Function numbers and register conventions are from the MOS API reference,
# https://agonplatform.github.io/agon-docs/mos/API/ - transcribed here rather
# than remembered, following the same rule EZ80.md applies to opcode tables.
#
# Only mos_load is used by shipped code. The handle-based calls are implemented
# so that a future streaming path has somewhere to land, and so that a build
# script that reaches for one fails on a wrong argument rather than on "not
# implemented" - but nothing calls them today.

# The two restarts and the two functions shipped code uses come from libagon,
# for the same reason the CP/M and Spectrum entry points come from their target
# modules: the address the emulator hooks and the address the code generator
# emits a RST to have to be the same one.
MOS_RST_OUTCHAR = AGON_MOS_OUTCHAR
MOS_RST_API = AGON_MOS_API
MOS_GETKEY = AGON_MOS_GETKEY
MOS_LOAD = AGON_MOS_LOAD

MOS_FOPEN = 0x0A        # HL=filename, C=mode -> A=handle (0 = failed)
MOS_FCLOSE = 0x0B       # C=handle
MOS_FREAD = 0x1A        # C=handle, HL=buffer, DE=count -> DE=bytes read
MOS_FLSEEK = 0x1C       # C=handle, HL=offset (low 24 bits), E=high byte

FA_READ = 0x01

#: Agon SRAM. A load outside this window would be discarded by the hardware or
#: land in flash; the emulator's memory is a plain bytearray that would happily
#: accept it, so the host is where that has to be caught.
AGON_RAM_LO = 0x040000
AGON_RAM_HI = 0x0C0000


class AgonHost(_StdinKeys):
    """eZ80 ADL-mode host implementing the MOS entry points we use.

    ``files`` is the SD card: a name -> bytes mapping, held in memory rather
    than on disk so that a test states exactly which bytes were served, no test
    needs a temporary directory, and there is no path-traversal surface to
    think about.
    """

    LOAD_ADDR = 0x040000

    def __init__(self, stdin: list[str] | None = None, ram_size: int = 0x800000,
                 files: dict[str, bytes] | None = None) -> None:
        self.cpu = Z80(adl=True, mem_size=ram_size)
        self.output: list[str] = []
        self.stdin = list(stdin or [])
        self.finished = False
        #: name -> contents. Lookup is case-insensitive, like FAT.
        self.files = {name.upper(): data for name, data in (files or {}).items()}
        #: handle -> [name, position]. Handle 0 means failure, so start at 1.
        self.handles: dict[int, list[Any]] = {}
        #: Bytes served from the card. bench.py reports this rather than
        #: pretending a hook that costs no T-states cost some.
        self.io_bytes = 0
        self.cpu.hooks[MOS_RST_OUTCHAR] = self._out_char
        self.cpu.hooks[MOS_RST_API] = self._mos_api
        self.cpu.sp = 0x0BFF00

    # --- memory helpers ------------------------------------------------------

    def read_cstring(self, addr: int, limit: int = 256) -> str:
        """A NUL-terminated filename out of emulated memory."""
        out = bytearray()
        for i in range(limit):
            byte = self.cpu.peek((addr + i) & self.cpu.amask)
            if byte == 0:
                return out.decode('latin-1')
            out.append(byte)
        raise Z80Error(f"unterminated string at {addr:06X}")

    def write_block(self, addr: int, data: bytes) -> None:
        """Copy ``data`` into emulated RAM, refusing anything out of bounds.

        Z80.load() grows its bytearray rather than failing, so without this a
        file read to a wrong address would look perfectly fine in the emulator
        and corrupt a real Agon. An emulator that cannot catch that is worse
        than no emulator, because it says the binary is good.
        """
        end = addr + len(data)
        if addr < AGON_RAM_LO or end > AGON_RAM_HI:
            raise Z80Error(
                f"load of {len(data)} bytes to {addr:06X} leaves Agon SRAM "
                f"({AGON_RAM_LO:06X}-{AGON_RAM_HI - 1:06X})")
        if end > len(self.cpu.mem):
            raise Z80Error(f"load to {addr:06X} past the end of memory")
        self.cpu.mem[addr:end] = data
        self.io_bytes += len(data)

    # --- MOS entry points ----------------------------------------------------

    def _out_char(self, cpu: Z80) -> bool:
        return self._emit_char(cpu)

    def _getkey(self, cpu: Z80) -> bool:
        return self._feed_key(cpu)

    def _load(self, cpu: Z80) -> bool:
        """mos_load: whole file to an address, one call, no handle to leak."""
        name = self.read_cstring(cpu.hl).upper()
        data = self.files.get(name)
        if data is None:
            cpu.a = 4                       # FR_NO_FILE
            return True
        if len(data) > cpu.bc:
            cpu.a = 5                       # would overrun the caller's buffer
            return True
        self.write_block(cpu.de, data)
        cpu.a = 0                           # FR_OK
        return True

    def _fopen(self, cpu: Z80) -> bool:
        name = self.read_cstring(cpu.hl).upper()
        if (cpu.c & FA_READ) == 0:
            raise Z80Error(f"mos_fopen mode {cpu.c:02X}: only reading is emulated")
        if name not in self.files:
            cpu.a = 0
            return True
        handle = next(i for i in range(1, 256) if i not in self.handles)
        self.handles[handle] = [name, 0]
        cpu.a = handle
        return True

    def _fclose(self, cpu: Z80) -> bool:
        if cpu.c == 0:
            self.handles.clear()
        else:
            self.handles.pop(cpu.c, None)
        cpu.a = len(self.handles)
        return True

    def _fread(self, cpu: Z80) -> bool:
        entry = self.handles.get(cpu.c)
        if entry is None:
            raise Z80Error(f"mos_fread on unopened handle {cpu.c}")
        name, pos = entry
        chunk = self.files[name][pos:pos + cpu.de]
        self.write_block(cpu.hl, chunk)
        entry[1] = pos + len(chunk)
        cpu.de = len(chunk)
        return True

    def _flseek(self, cpu: Z80) -> bool:
        entry = self.handles.get(cpu.c)
        if entry is None:
            raise Z80Error(f"mos_flseek on unopened handle {cpu.c}")
        entry[1] = (cpu.hl & 0xFFFFFF) | (cpu.e << 24)
        cpu.a = 0
        return True

    #: Dispatch on A. Anything absent raises, which is the property the module
    #: docstring promises and the reason a typo fails loudly.
    _API: ClassVar[dict[int, Callable[[AgonHost, Z80], bool]]] = {
        MOS_GETKEY: _getkey,
        MOS_LOAD: _load,
        MOS_FOPEN: _fopen,
        MOS_FCLOSE: _fclose,
        MOS_FREAD: _fread,
        MOS_FLSEEK: _flseek,
    }

    def _mos_api(self, cpu: Z80) -> bool:
        handler = self._API.get(cpu.a)
        if handler is None:
            raise Z80Error(f"unimplemented MOS API function {cpu.a:02X}")
        if handler(self, cpu):
            cpu.pc = cpu._pop()
        return True

    def run(self, image: bytes, max_cycles: int = 2_000_000_000) -> str:
        self.cpu.load(self.LOAD_ADDR, image)
        self.cpu.pc = self.LOAD_ADDR
        # MOS calls the program; returning drops onto a HALT.
        self.cpu.poke(0x0BFFF0, 0x76)
        self.cpu.sp = 0x0BFF00
        for k in range(3):
            self.cpu.poke(0x0BFF00 + k, (0x0BFFF0 >> (8 * k)) & 0xFF)
        self.cpu.run(max_cycles=max_cycles)
        return "".join(self.output)


def run_agon(
    image: bytes,
    stdin: list[str] | None = None,
    max_cycles: int = 2_000_000_000,
    files: dict[str, bytes] | None = None,
) -> tuple[str, AgonHost]:
    host = AgonHost(stdin=stdin, files=files)
    return host.run(image, max_cycles=max_cycles), host
