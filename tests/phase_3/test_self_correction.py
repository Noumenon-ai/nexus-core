from services.self_correction import detect_self_correction


def test_numeric_month_day_correction():
    assert detect_self_correction('take Kia to mechanic june 2 no june 4') == (
        'date', 'June 2', 'June 4',
    )


def test_month_day_correction_with_comma_and_wait():
    assert detect_self_correction('remind me may 5, wait no may 7 to call') == (
        'date', 'May 5', 'May 7',
    )


def test_time_correction_uppercases_meridiem():
    assert detect_self_correction('set it for 5pm actually 6pm') == (
        'time', '5 PM', '6 PM',
    )


def test_time_correction_with_colon():
    assert detect_self_correction('alarm 5:30pm no 6:15pm') == (
        'time', '5:30 PM', '6:15 PM',
    )


def test_weekday_correction():
    assert detect_self_correction('meeting monday no tuesday') == (
        'date', 'Monday', 'Tuesday',
    )


def test_relative_correction():
    assert detect_self_correction('do it today nope tomorrow') == (
        'date', 'Today', 'Tomorrow',
    )


def test_latest_correction_wins_when_multiple():
    assert detect_self_correction('june 2 no june 4 wait no june 6') == (
        'date', 'June 4', 'June 6',
    )


def test_no_correction_returns_none():
    assert detect_self_correction('remind me june 4 to take Kia') is None


def test_no_rush_is_not_a_correction():
    # "no" not followed by a same-kind token must not trigger.
    assert detect_self_correction('remind me at 5pm, no rush, tomorrow') is None


def test_not_keeps_first_value():
    # "X not Y" keeps X, rejects Y -> reverse direction, X is the new value.
    assert detect_self_correction('june 4 not june 2') == (
        'date', 'June 2', 'June 4',
    )


def test_same_value_is_not_a_correction():
    assert detect_self_correction('june 4 no june 4') is None


# -------- widened phrasings -------------------------------------------------

def test_scratch_that_move_it_to_ordinal():
    assert detect_self_correction(
        'remind me to take Kia june 2 scratch that move it to the 4th'
    ) == ('date', 'June 2', 'the 4th')


def test_change_it_to_time():
    assert detect_self_correction('set the alarm 5pm change it to 6pm') == (
        'time', '5 PM', '6 PM',
    )


def test_make_it_correction():
    assert detect_self_correction('june 2, make it june 4') == (
        'date', 'June 2', 'June 4',
    )


def test_lets_do_with_comma():
    assert detect_self_correction('call dad monday, actually lets do wednesday') == (
        'date', 'Monday', 'Wednesday',
    )


def test_cross_representation_weekday_to_ordinal():
    assert detect_self_correction('move it monday no the 4th') == (
        'date', 'Monday', 'the 4th',
    )


def test_bare_ordinal_old_and_new():
    assert detect_self_correction('the 2nd no the 4th') == (
        'date', 'the 2nd', 'the 4th',
    )


def test_time_range_is_not_a_correction():
    # "from 5pm to 6pm" is a range, not a self-correction.
    assert detect_self_correction('remind me from 5pm to 6pm') is None


def test_instead_of_keeps_first_value():
    # "X instead of Y" keeps X, rejects Y -> reverse direction.
    assert detect_self_correction('june 4 instead of june 2') == (
        'date', 'June 2', 'June 4',
    )


def test_rather_than_keeps_first_value():
    assert detect_self_correction('lets meet wednesday rather than monday') == (
        'date', 'Monday', 'Wednesday',
    )


def test_but_not_is_not_a_correction():
    # "but" is real content in the gap -> blocks the pair.
    assert detect_self_correction('i can do monday but not friday') is None


def test_and_is_not_a_connector():
    assert detect_self_correction('remind me monday and tuesday') is None


def test_two_separate_dates_in_prose_not_fired():
    assert detect_self_correction(
        'see you friday, have a great weekend, talk monday'
    ) is None


def test_chain_across_phrasings_latest_wins():
    assert detect_self_correction(
        'june 2 no june 4 scratch that the 6th'
    ) == ('date', 'June 4', 'the 6th')
