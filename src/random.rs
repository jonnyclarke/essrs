//! Module providing helper functions used for the random sampling of walkers used by differential move algorithm.

use rand::{Rng, RngCore};

/// This is used to sample a random index that does NOT equal the index of the current walker.
/// Given we are running `N_WALKERS` as the ensemble.
/// Differential move requires two further walkers to define the direction vector.
/// We sample the first by sampling integers in the range [0, N_WALKERS - 1) and, if the selected index equals our walker's index (given by `not_this`) we add 1 to avoid re-sampling that value.
pub fn get_rn_not(rng: &mut dyn RngCore, max_index: usize, not_this: usize) -> usize {
    let mut random_number: usize = rng.random_range(0..max_index - 1);

    if random_number >= not_this {
        random_number += 1
    }

    random_number
}

/// This is the follow up function needed to sample the 2nd walker for construction of the direction vector.
/// The logic is identical aside from we now provide two indexes to avoid by virtue of adding 1 to the result.
pub fn get_rn_not_or(
    rng: &mut dyn RngCore,
    max_index: usize,
    not_this: usize,
    or_this: usize,
) -> usize {
    let (low, high) = if not_this < or_this {
        (not_this, or_this)
    } else {
        (or_this, not_this)
    };
    let mut random_number: usize = rng.random_range(0..max_index - 2);

    if random_number >= low {
        random_number += 1
    }

    if random_number >= high {
        random_number += 1
    }

    random_number
}

#[cfg(test)]
mod tests {
    use rand::rng;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(5, 0)]
    #[case(300, 136)]
    fn test_get_rn_not(#[case] n: usize, #[case] not: usize) {
        let mut rand = rng();
        assert_ne!(get_rn_not(&mut rand, n, not), not);
    }

    #[rstest]
    #[case(2, 0, 1)]
    #[case(2, 1, 0)]
    fn test_get_rn_not_one_option(#[case] n: usize, #[case] not: usize, #[case] correct: usize) {
        let mut rand = rng();
        assert_eq!(get_rn_not(&mut rand, n, not), correct);
    }

    #[rstest]
    #[case(5, 0, 1)]
    #[case(50, 13, 23)]
    fn test_get_rn_not_or(#[case] n: usize, #[case] not: usize, #[case] and_not: usize) {
        let mut rand = rng();
        assert_ne!(get_rn_not_or(&mut rand, n, not, and_not), not);
        assert_ne!(get_rn_not_or(&mut rand, n, not, and_not), and_not);
    }

    #[rstest]
    #[case(3, 0, 1, 2)]
    #[case(3, 1, 2, 0)]
    #[case(3, 2, 0, 1)]
    fn test_get_rn_not_or_one_option(
        #[case] n: usize,
        #[case] not: usize,
        #[case] and_not: usize,
        #[case] correct: usize,
    ) {
        let mut rand = rng();
        assert_eq!(get_rn_not_or(&mut rand, n, not, and_not), correct);
    }
}
