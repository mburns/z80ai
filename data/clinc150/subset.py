#!/usr/bin/env python3
"""
Turn CLINC150 into a training file for a chosen set of intents.

CLINC150 labels *intents*, not replies, so the mapping from intent to the string
the model should emit is ours to author - that is what a RECIPE is.

There are two registers here, and they want opposite things:

  A character decoder spells its answer one output neuron at a time, so replies
  must be terse and share letters; every distinct character costs 128 weights.
  That is what RECIPES is for, and what the shipped `smalltalk` example uses.

  A phrasebook emits an *index* into a table of replies held on an SD card, so
  the reply text costs the model nothing at all.  That is what REPLIES is for.
  Full sentences, one per intent, and the model never has to spell any of them.

    python data/clinc150/subset.py --recipe smalltalk > out.txt
    python data/clinc150/subset.py --recipe clinc150 > all.txt
    python data/clinc150/subset.py --recipe clinc-router > router.txt
    python data/clinc150/subset.py --domain banking > banking.txt
    python data/clinc150/subset.py --list           # every intent and its domain
    python data/lint.py out.txt --strict

150 intents is far more than a 2-bit character model can learn, so a RECIPE
picks a handful.  A phrasebook plus one expert per domain can carry all of them;
`--domain` emits the training file for one such expert, and `--recipe
clinc-router` emits the file that teaches the router which expert to page in.

`oos` is CLINC's own out-of-scope class - 1,200 utterances of things a system
cannot answer - which is a much better catch-all than invented nonsense.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

DATA = Path(__file__).with_name('data_full.json.gz')

#: name -> {intent: reply}.  Add recipes here rather than editing one in place;
#: a shipped example depends on its recipe staying put.
RECIPES: dict[str, dict[str, str]] = {
    # The chatbot register: what someone types at a toy that talks back.
    'smalltalk': {
        'greeting': 'HI',
        'goodbye': 'BYE',
        'thank_you': 'NO PROB',
        'yes': 'OK',
        'no': 'NOPE',
        'maybe': 'MAYBE',
        'are_you_a_bot': 'IM A BOT',
        'what_is_your_name': 'IM CHAT',
        'how_old_are_you': 'OLD',
        'where_are_you_from': 'A ROM CHIP',
        'who_made_you': 'A HUMAN',
        'what_are_your_hobbies': 'BITS',
        'do_you_have_pets': 'NO PETS',
        'tell_joke': 'HA HA',
        'fun_fact': 'HMM',
        'meaning_of_life': 'WHO KNOWS',
        'what_can_i_ask_you': 'ASK ME',
        'repeat': 'AGAIN',
        'oos': 'IDK',
    },
}

#: CLINC's ten domains, fifteen intents each.  The grouping is **not** in
#: data_full.json.gz - that file is a flat [utterance, intent] list - so it is
#: transcribed here from Larson et al., "An Evaluation Dataset for Intent
#: Classification and Out-of-Scope Prediction", EMNLP 2019, section 3.
#: https://github.com/clinc/oos-eval (CC BY 3.0)
DOMAINS: dict[str, list[str]] = {
    'banking': [
        'freeze_account', 'routing', 'pin_change', 'bill_due', 'pay_bill',
        'account_blocked', 'interest_rate', 'min_payment', 'bill_balance',
        'transfer', 'order_checks', 'balance', 'spending_history',
        'transactions', 'report_fraud',
    ],
    'credit_cards': [
        'credit_limit', 'improve_credit_score', 'credit_score', 'card_declined',
        'credit_limit_change', 'damaged_card', 'replacement_card_duration',
        'new_card', 'rewards_balance', 'report_lost_card', 'apr',
        'expiration_date', 'international_fees', 'redeem_rewards',
        'application_status',
    ],
    'kitchen_and_dining': [
        'ingredient_substitution', 'cook_time', 'recipe', 'restaurant_reviews',
        'restaurant_reservation', 'meal_suggestion', 'restaurant_suggestion',
        'cancel_reservation', 'ingredients_list', 'nutrition_info', 'calories',
        'how_busy', 'accept_reservations', 'food_last', 'confirm_reservation',
    ],
    'home': [
        'shopping_list', 'shopping_list_update', 'next_song', 'play_music',
        'update_playlist', 'todo_list', 'todo_list_update', 'calendar',
        'calendar_update', 'what_song', 'order', 'order_status', 'reminder',
        'reminder_update', 'smart_home',
    ],
    'auto_and_commute': [
        'current_location', 'oil_change_when', 'oil_change_how', 'uber',
        'traffic', 'tire_pressure', 'schedule_maintenance', 'gas', 'mpg',
        'distance', 'directions', 'last_maintenance', 'gas_type', 'tire_change',
        'jump_start',
    ],
    'travel': [
        'plug_type', 'travel_notification', 'translate', 'flight_status',
        'international_visa', 'timezone', 'exchange_rate', 'travel_suggestion',
        'travel_alert', 'vaccines', 'lost_luggage', 'book_flight', 'book_hotel',
        'carry_on', 'car_rental',
    ],
    'utility': [
        'weather', 'alarm', 'date', 'find_phone', 'share_location', 'timer',
        'make_call', 'calculator', 'definition', 'measurement_conversion',
        'flip_coin', 'spelling', 'time', 'roll_dice', 'text',
    ],
    'work': [
        'pto_request_status', 'next_holiday', 'insurance_change', 'insurance',
        'meeting_schedule', 'payday', 'taxes', 'income', 'rollover_401k',
        'pto_balance', 'pto_request', 'w2', 'schedule_meeting',
        'direct_deposit', 'pto_used',
    ],
    'small_talk': [
        'greeting', 'goodbye', 'tell_joke', 'where_are_you_from',
        'how_old_are_you', 'what_is_your_name', 'who_made_you',
        'what_are_your_hobbies', 'do_you_have_pets', 'are_you_a_bot',
        'meaning_of_life', 'fun_fact', 'what_can_i_ask_you',
        'who_do_you_work_for', 'change_ai_name',
    ],
    'meta': [
        'change_speed', 'user_name', 'whisper_mode', 'yes', 'no', 'maybe',
        'change_language', 'repeat', 'change_accent', 'cancel', 'sync_device',
        'change_user_name', 'change_volume', 'reset_settings', 'thank_you',
    ],
}

#: intent -> the full-sentence reply a phrasebook would hold.  Ours, not
#: CLINC's; the dataset ships utterances and labels only.  These are never
#: spelled by the model, so they are free to be sentences - but they must stay
#: under libdata.MAX_RESPONSE_LEN or parse_pair truncates them at a word
#: boundary, and they must stay distinct or two intents collapse into one class.
REPLIES: dict[str, str] = {
    # banking
    'freeze_account': 'I FROZE YOUR ACCOUNT',
    'routing': 'YOUR ROUTING NUMBER IS ON YOUR CHECKS',
    'pin_change': 'I CAN CHANGE YOUR PIN',
    'bill_due': 'THAT BILL IS DUE ON THE FIRST',
    'pay_bill': 'PAYING THAT BILL NOW',
    'account_blocked': 'YOUR ACCOUNT IS BLOCKED',
    'interest_rate': 'YOUR INTEREST RATE IS TWO PERCENT',
    'min_payment': 'THE MINIMUM PAYMENT IS TWENTY FIVE',
    'bill_balance': 'YOUR BILL BALANCE IS NINETY DOLLARS',
    'transfer': 'TRANSFERRING BETWEEN YOUR ACCOUNTS',
    'order_checks': 'I ORDERED MORE CHECKS',
    'balance': 'YOUR BALANCE IS FOUR HUNDRED DOLLARS',
    'spending_history': 'HERE IS WHAT YOU SPENT',
    'transactions': 'HERE ARE YOUR RECENT TRANSACTIONS',
    'report_fraud': 'I REPORTED THE FRAUD',
    # credit_cards
    'credit_limit': 'YOUR CREDIT LIMIT IS FIVE THOUSAND',
    'improve_credit_score': 'PAY ON TIME TO IMPROVE YOUR SCORE',
    'credit_score': 'YOUR CREDIT SCORE IS SEVEN TWENTY',
    'card_declined': 'THAT CARD WAS DECLINED',
    'credit_limit_change': 'I REQUESTED A LIMIT CHANGE',
    'damaged_card': 'I WILL REPLACE THE DAMAGED CARD',
    'replacement_card_duration': 'A NEW CARD TAKES ABOUT A WEEK',
    'new_card': 'I ORDERED YOU A NEW CARD',
    'rewards_balance': 'YOU HAVE TWO THOUSAND REWARD POINTS',
    'report_lost_card': 'I REPORTED YOUR CARD LOST',
    'apr': 'YOUR APR IS FOURTEEN PERCENT',
    'expiration_date': 'THAT CARD EXPIRES NEXT MARCH',
    'international_fees': 'THERE IS A THREE PERCENT FOREIGN FEE',
    'redeem_rewards': 'REDEEMING YOUR REWARDS NOW',
    'application_status': 'YOUR APPLICATION IS STILL PENDING',
    # kitchen_and_dining
    'ingredient_substitution': 'YOU CAN SUBSTITUTE BUTTER FOR OIL',
    'cook_time': 'COOK IT FOR ABOUT FORTY MINUTES',
    'recipe': 'HERE IS A RECIPE FOR THAT',
    'restaurant_reviews': 'THAT PLACE HAS FOUR STARS',
    'restaurant_reservation': 'I BOOKED YOUR TABLE',
    'meal_suggestion': 'TRY PASTA TONIGHT',
    'restaurant_suggestion': 'THERE IS A GOOD DINER NEARBY',
    'cancel_reservation': 'I CANCELLED YOUR RESERVATION',
    'ingredients_list': 'HERE IS WHAT YOU WILL NEED',
    'nutrition_info': 'HERE IS THE NUTRITION INFORMATION',
    'calories': 'THAT IS ABOUT THREE HUNDRED CALORIES',
    'how_busy': 'IT LOOKS BUSY RIGHT NOW',
    'accept_reservations': 'YES THEY TAKE RESERVATIONS',
    'food_last': 'THAT KEEPS FOR ABOUT FOUR DAYS',
    'confirm_reservation': 'YOUR RESERVATION IS CONFIRMED',
    # home
    'shopping_list': 'HERE IS YOUR SHOPPING LIST',
    'shopping_list_update': 'I UPDATED YOUR SHOPPING LIST',
    'next_song': 'SKIPPING TO THE NEXT SONG',
    'play_music': 'PLAYING MUSIC NOW',
    'update_playlist': 'I UPDATED YOUR PLAYLIST',
    'todo_list': 'HERE IS YOUR TODO LIST',
    'todo_list_update': 'I UPDATED YOUR TODO LIST',
    'calendar': 'HERE IS WHAT IS ON YOUR CALENDAR',
    'calendar_update': 'I UPDATED YOUR CALENDAR',
    'what_song': 'THIS SONG IS CLAIR DE LUNE',
    'order': 'I PLACED THAT ORDER',
    'order_status': 'YOUR ORDER SHIPS TOMORROW',
    'reminder': 'HERE ARE YOUR REMINDERS',
    'reminder_update': 'I SET THAT REMINDER',
    'smart_home': 'DONE THE LIGHTS ARE SET',
    # auto_and_commute
    'current_location': 'YOU ARE ON MAIN STREET',
    'oil_change_when': 'YOUR OIL IS DUE IN TWO HUNDRED MILES',
    'oil_change_how': 'DRAIN THE PAN THEN REPLACE THE FILTER',
    'uber': 'YOUR RIDE IS FIVE MINUTES AWAY',
    'traffic': 'TRAFFIC IS HEAVY ON YOUR ROUTE',
    'tire_pressure': 'YOUR TIRES ARE AT THIRTY PSI',
    'schedule_maintenance': 'I SCHEDULED YOUR SERVICE',
    'gas': 'YOU HAVE A QUARTER TANK LEFT',
    'mpg': 'YOU ARE GETTING THIRTY MILES A GALLON',
    'distance': 'THAT IS TWELVE MILES AWAY',
    'directions': 'HEAD NORTH THEN TURN RIGHT',
    'last_maintenance': 'YOUR LAST SERVICE WAS IN JUNE',
    'gas_type': 'THAT CAR TAKES REGULAR UNLEADED',
    'tire_change': 'LOOSEN THE NUTS BEFORE YOU JACK IT UP',
    'jump_start': 'RED TO POSITIVE THEN BLACK TO GROUND',
    # travel
    'plug_type': 'YOU WILL NEED A TYPE C ADAPTER',
    'travel_notification': 'I TOLD YOUR BANK YOU ARE TRAVELLING',
    'translate': 'THAT TRANSLATES TO BONJOUR',
    'flight_status': 'YOUR FLIGHT IS ON TIME',
    'international_visa': 'YOU WILL NEED A VISA FOR THAT TRIP',
    'timezone': 'THAT PLACE IS THREE HOURS AHEAD',
    'exchange_rate': 'ONE DOLLAR IS ABOUT NINETY CENTS',
    'travel_suggestion': 'LISBON IS LOVELY THIS TIME OF YEAR',
    'travel_alert': 'THERE IS AN ALERT FOR THAT AREA',
    'vaccines': 'YOU WILL NEED A YELLOW FEVER SHOT',
    'lost_luggage': 'I FILED A REPORT FOR YOUR BAG',
    'book_flight': 'I BOOKED YOUR FLIGHT',
    'book_hotel': 'I BOOKED YOUR HOTEL',
    'carry_on': 'ONE BAG AND ONE PERSONAL ITEM',
    'car_rental': 'I RESERVED YOU A RENTAL CAR',
    # utility
    'weather': 'IT IS SIXTY DEGREES AND CLEAR',
    'alarm': 'I SET YOUR ALARM',
    'date': 'TODAY IS THE TWELFTH',
    'find_phone': 'I AM RINGING YOUR PHONE NOW',
    'share_location': 'I SHARED YOUR LOCATION',
    'timer': 'I STARTED YOUR TIMER',
    'make_call': 'CALLING NOW',
    'calculator': 'THAT COMES TO FORTY TWO',
    'definition': 'IT MEANS A SMALL ROUND OBJECT',
    'measurement_conversion': 'THAT IS ABOUT TWO AND A HALF CENTIMETRES',
    'flip_coin': 'IT CAME UP HEADS',
    'spelling': 'THAT IS SPELLED WITH TWO LETTER LS',
    'time': 'IT IS HALF PAST THREE',
    'roll_dice': 'YOU ROLLED A FOUR',
    'text': 'I SENT THAT TEXT',
    # work
    'pto_request_status': 'YOUR TIME OFF REQUEST WAS APPROVED',
    'next_holiday': 'THE NEXT HOLIDAY IS IN NOVEMBER',
    'insurance_change': 'I UPDATED YOUR INSURANCE PLAN',
    'insurance': 'YOU ARE ON THE STANDARD HEALTH PLAN',
    'meeting_schedule': 'YOU HAVE TWO MEETINGS TODAY',
    'payday': 'YOU GET PAID ON FRIDAY',
    'taxes': 'YOUR TAXES ARE DUE IN APRIL',
    'income': 'YOU EARNED FIFTY THOUSAND LAST YEAR',
    'rollover_401k': 'I CAN ROLL OVER YOUR RETIREMENT PLAN',
    'pto_balance': 'YOU HAVE TEN DAYS OF TIME OFF LEFT',
    'pto_request': 'I REQUESTED YOUR TIME OFF',
    'w2': 'YOUR TAX FORM WAS MAILED IN JANUARY',
    'schedule_meeting': 'I PUT THAT MEETING ON THE CALENDAR',
    'direct_deposit': 'I SET UP YOUR DIRECT DEPOSIT',
    'pto_used': 'YOU HAVE USED FIVE DAYS SO FAR',
    # small_talk
    'greeting': 'HELLO THERE',
    'goodbye': 'GOODBYE FOR NOW',
    'tell_joke': 'WHY DID THE CHICKEN CROSS THE ROAD',
    'where_are_you_from': 'I LIVE ON A ROM CHIP',
    'how_old_are_you': 'I AM OLDER THAN I LOOK',
    'what_is_your_name': 'MY NAME IS CHAT',
    'who_made_you': 'A HUMAN WROTE ME',
    'what_are_your_hobbies': 'I MOSTLY COUNT IN BINARY',
    'do_you_have_pets': 'NO PETS JUST TRANSISTORS',
    'are_you_a_bot': 'YES I AM A BOT',
    'meaning_of_life': 'NOBODY HAS TOLD ME YET',
    'fun_fact': 'HONEY NEVER SPOILS',
    'what_can_i_ask_you': 'ASK ME ABOUT ALMOST ANYTHING',
    'who_do_you_work_for': 'I WORK FOR YOU',
    'change_ai_name': 'I WILL ANSWER TO THAT NAME',
    # meta
    'change_speed': 'I WILL TALK AT THAT SPEED',
    'user_name': 'I HAVE YOUR NAME ON FILE',
    'whisper_mode': 'I WILL WHISPER FROM NOW ON',
    'yes': 'GOT IT',
    'no': 'UNDERSTOOD',
    'maybe': 'I WILL TAKE THAT AS A MAYBE',
    'change_language': 'I SWITCHED LANGUAGES',
    'repeat': 'LET ME SAY THAT AGAIN',
    'change_accent': 'I CHANGED MY ACCENT',
    'cancel': 'CANCELLED',
    'sync_device': 'I SYNCED YOUR DEVICE',
    'change_user_name': 'I CHANGED YOUR USER NAME',
    'change_volume': 'I CHANGED THE VOLUME',
    'reset_settings': 'I RESET YOUR SETTINGS',
    'thank_you': 'HAPPY TO HELP',
    # CLINC's own out-of-scope class.
    'oos': 'I DO NOT KNOW THAT ONE',
}

#: intent -> domain, inverted once so the router recipe and --domain agree.
DOMAIN_OF: dict[str, str] = {
    intent: domain for domain, intents in DOMAINS.items() for intent in intents
}

# A reply that appears twice silently merges two intents into one class, and a
# domain list that drifts from REPLIES makes --domain quietly emit less than it
# claims. Both are the kind of mistake that trains fine and answers wrong, so
# they fail at import rather than at the end of a training run.
assert len(set(REPLIES.values())) == len(REPLIES), \
    "two intents share a reply: " + str(sorted(
        r for r in set(REPLIES.values())
        if list(REPLIES.values()).count(r) > 1))
assert set(DOMAIN_OF) | {'oos'} == set(REPLIES), \
    f"DOMAINS and REPLIES disagree: {sorted(set(DOMAIN_OF) ^ (set(REPLIES) - {'oos'}))}"


def load() -> list[tuple[str, str]]:
    """Every (utterance, intent) pair, across all of CLINC's splits.

    The published train/val/test split is not used: feedme does its own, by
    unique query, and mixing the splits here would only hide that.
    """
    with gzip.open(DATA, 'rt') as fh:
        data = json.load(fh)
    return [(q, label) for rows in data.values() for q, label in rows]


def build(recipe: dict[str, str], seed: int = 0, cap: int | None = None,
          oos_cap: int | float | None = None) -> list[tuple[str, str]]:
    """Balanced (query, reply) pairs for one recipe.

    Every in-scope intent ships exactly 150 utterances but `oos` ships 1,200,
    so without a cap the catch-all would be a third of the data and become the
    model's default answer.

    ``oos_cap`` overrides that for the router recipe, where fifteen intents
    collapse into one reply: capping `oos` at the per-*intent* size would leave
    the catch-all outnumbered fifteen to one and drop it below the class share
    data/lint.py will accept.
    """
    by_intent: dict[str, set[str]] = {}
    for query, intent in load():
        if intent in recipe:
            by_intent.setdefault(intent, set()).add(query)

    missing = set(recipe) - set(by_intent)
    if missing:
        raise SystemExit(f"no such intent(s): {sorted(missing)}")

    in_scope = [len(v) for k, v in by_intent.items() if k != 'oos']
    if cap is None:
        cap = min(in_scope) if in_scope else max(map(len, by_intent.values()))

    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    for intent in sorted(by_intent):                 # sorted: reproducible
        queries = sorted(by_intent[intent])
        limit = oos_cap if intent == 'oos' and oos_cap is not None else cap
        if len(queries) > limit:
            queries = rng.sample(queries, limit)
        pairs.extend((q.upper(), recipe[intent]) for q in queries)

    rng.shuffle(pairs)
    return pairs


def resolve(name: str | None,
            domain: str | None) -> tuple[dict[str, str], int | float | None]:
    """The (recipe, oos_cap) a set of CLI options asks for."""
    if domain:
        # One expert's slice, plus the catch-all so it can decline politely.
        intents = DOMAINS[domain] + ['oos']
        return {i: REPLIES[i] for i in intents}, None
    if name == 'clinc150':
        return dict(REPLIES), None
    if name == 'clinc-router':
        # The router learns the *domain*, not the intent, so fifteen intents
        # share one reply and each domain arrives fifteen times over. Capping
        # `oos` per-intent alongside them leaves the catch-all at 0.7% of the
        # data - under what data/lint.py will accept, and far under what it
        # takes to learn. Uncapped it is 5%, which is about right.
        recipe = {i: DOMAIN_OF[i].upper().replace('_', ' ') for i in DOMAIN_OF}
        recipe['oos'] = 'IDK'
        return recipe, float('inf')
    return RECIPES[name], None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--recipe', default='smalltalk',
                        choices=[*sorted(RECIPES), 'clinc150', 'clinc-router'])
    parser.add_argument('--domain', choices=sorted(DOMAINS),
                        help='Emit one domain expert\'s training file instead '
                             'of a recipe (fifteen intents plus the catch-all)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cap', type=int, default=None,
                        help='Max utterances per intent (default: the smallest '
                             'in-scope intent, so the classes come out balanced)')
    parser.add_argument('--list', action='store_true',
                        help='List every intent in the dataset and exit')
    args = parser.parse_args()

    if args.list:
        intents = sorted({intent for _, intent in load()})
        print(f"# {len(intents)} intents", file=sys.stderr)
        for intent in intents:
            print(f"{intent}\t{DOMAIN_OF.get(intent, '-')}")
        return

    recipe, oos_cap = resolve(args.recipe, args.domain)
    pairs = build(recipe, args.seed, args.cap, oos_cap)
    label = f"domain '{args.domain}'" if args.domain else f"recipe '{args.recipe}'"

    print(f"# CLINC150, {label}, seed {args.seed}.")
    print("# Generated by data/clinc150/subset.py - do not edit by hand.")
    print(f"# {len(pairs)} pairs over {len(set(recipe.values()))} replies.")
    print("# Source: https://github.com/clinc/oos-eval (CC BY 3.0), Larson et al.,")
    print("# EMNLP 2019. Utterances are uppercased; replies are ours.")
    for query, reply in pairs:
        print(f"{query}|{reply}")


if __name__ == '__main__':
    main()
