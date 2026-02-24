use rand::{Rng, RngCore};

pub fn get_rn_not(rng: &mut dyn RngCore, max_index: usize, not_this: usize) -> usize {
    let mut random_number: usize = rng.random_range(0..max_index - 1);

    if random_number >= not_this {
        random_number += 1
    }

    random_number
}

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
