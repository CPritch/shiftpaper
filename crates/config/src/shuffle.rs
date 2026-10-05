//! The order a shuffled slideshow shows its images in.

/// The order to show `len` images in on the `round`th time through them.
/// The same round always gives the same order, so it can be worked out
/// from the clock with nothing saved. Images from the last third of one
/// round are kept out of the first third of the next, so none comes back
/// too soon.
pub fn order(round: u64, len: usize) -> Vec<usize> {
    // With two images, the only way never to repeat one is to alternate.
    if len <= 2 {
        return (0..len).collect();
    }
    let mut order = permutation(round, len);
    if round == 0 {
        return order;
    }
    let third = len / 3;
    // Only the first two thirds are ever rearranged, so the last third of
    // the previous round is the same as in its plain permutation.
    let previous = permutation(round - 1, len);
    let recent = &previous[len - third..];
    // Swap each recent image at the start for one from the middle third
    // that isn't recent. There are always enough: of the third that are
    // recent, any at the start aren't taking up room in the middle.
    let mut middle = third..len - third;
    for i in 0..third {
        if recent.contains(&order[i]) {
            let j = middle
                .find(|&j| !recent.contains(&order[j]))
                .expect("the middle third has room");
            order.swap(i, j);
        }
    }
    order
}

/// A random-looking ordering of 0..len, seeded by `round`.
fn permutation(round: u64, len: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    let mut state = round;
    // Fisher-Yates: swap each position with a random one at or before it.
    for i in (1..len).rev() {
        let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

/// A small, well known pseudo-random number generator. Plenty for picking
/// wallpapers, and it saves a dependency.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_round_shows_every_image_once() {
        for round in 0..50 {
            let mut seen = order(round, 9);
            seen.sort();
            assert_eq!(seen, (0..9).collect::<Vec<_>>(), "round {round}");
        }
    }

    #[test]
    fn rounds_are_repeatable_and_differ() {
        assert_eq!(order(7, 9), order(7, 9));
        assert_ne!(order(7, 9), order(8, 9));
    }

    #[test]
    fn no_image_shows_twice_in_a_row_between_rounds() {
        for len in 2..8 {
            for round in 1..500 {
                let before = order(round - 1, len);
                assert_ne!(
                    order(round, len)[0],
                    before[len - 1],
                    "len {len}, round {round}"
                );
            }
        }
    }

    #[test]
    fn images_wait_at_least_a_third_of_the_slideshow_to_come_back() {
        for len in 3..13 {
            let shown: Vec<usize> = (0..300).flat_map(|round| order(round, len)).collect();
            for (i, image) in shown.iter().enumerate() {
                let next = shown[i + 1..].iter().position(|other| other == image);
                if let Some(gap) = next {
                    assert!(gap >= len / 3, "len {len}: {image} back after {gap}");
                }
            }
        }
    }

    #[test]
    fn tiny_slideshows_work() {
        assert_eq!(order(3, 1), [0]);
        assert!(order(3, 0).is_empty());
    }
}
