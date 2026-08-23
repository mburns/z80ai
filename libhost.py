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

# Entry points and memory maps come from the target modules rather than being
# restated: the emulator's idea of where BDOS sits and the code generator's have
# to agree, and the surest way to guarantee that is to have only one of them.
from libcpm import BDOS, CPM_CMDLINE, TPA, TPA_TOP
from libz80emu import Z80, Z80Error
from libzx import ORG_ADDR as ZX_DEFAULT_ORG
from libzx import ZX_CHAN_OPEN, ZX_CLS, ZX_KEY_INPUT, ZX_PRINT_A, ZX_RAM_TOP

#: Where a transient program's stack starts, a little below the BDOS. Real CP/M
#: hands over whatever the CCP was using; anywhere clear of the image will do.
CPM_STACK = 0xE000

# --- CP/M --------------------------------------------------------------------


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


class ZXHost:
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
        self.output.append(chr(cpu.a))
        cpu.pc = cpu._pop()
        return True

    def _key_input(self, cpu: Z80) -> bool:
        if not self.stdin:
            self.finished = True
            cpu.halted = True
            return True
        line = self.stdin[0]
        if line == "":
            self.stdin.pop(0)
            cpu.a = 13
        else:
            self.stdin[0] = line[1:]
            cpu.a = ord(line[0])
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


# --- Agon / eZ80 MOS ---------------------------------------------------------

MOS_RST_OUTCHAR = 0x10
MOS_RST_API = 0x08
MOS_GETKEY = 0x00


class AgonHost:
    """eZ80 ADL-mode host implementing the two MOS entry points we use."""

    LOAD_ADDR = 0x040000

    def __init__(self, stdin: list[str] | None = None, ram_size: int = 0x800000) -> None:
        self.cpu = Z80(adl=True, mem_size=ram_size)
        self.output: list[str] = []
        self.stdin = list(stdin or [])
        self.finished = False
        self.cpu.hooks[MOS_RST_OUTCHAR] = self._out_char
        self.cpu.hooks[MOS_RST_API] = self._mos_api
        self.cpu.sp = 0x0BFF00

    def _out_char(self, cpu: Z80) -> bool:
        self.output.append(chr(cpu.a))
        cpu.pc = cpu._pop()
        return True

    def _mos_api(self, cpu: Z80) -> bool:
        fn = cpu.a
        if fn == MOS_GETKEY:
            if not self.stdin:
                self.finished = True
                cpu.halted = True
                return True
            line = self.stdin[0]
            if line == "":
                self.stdin.pop(0)
                cpu.a = 13
            else:
                self.stdin[0] = line[1:]
                cpu.a = ord(line[0])
        else:
            raise Z80Error(f"unimplemented MOS API function {fn:02X}")
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
) -> tuple[str, AgonHost]:
    host = AgonHost(stdin=stdin)
    return host.run(image, max_cycles=max_cycles), host
