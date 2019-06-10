use std::fs::File;
use std::io::Write;

use rand::distributions::{Distribution, LogNormal};
use rand::{FromEntropy, XorShiftRng};

pub fn create_dataset<F>(mut f: F, count: usize) -> Box<[f32]>
where
    F: FnMut() -> f32,
{
    let mut dataset = Vec::with_capacity(count);
    for _ in 0..count {
        dataset.push(f());
    }
    dataset.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dataset.into_boxed_slice()
}

pub fn lognorm_dist(count: usize) -> Box<[f32]> {
    let mut rng = XorShiftRng::from_entropy();
    let lognorm = LogNormal::new(0.0, 0.25);
    create_dataset(|| lognorm.sample(&mut rng) as f32, count)
}

fn main() {
    let count: usize = 10000;
    let data = lognorm_dist(count);

    let mut file = File::create("dataset").expect("Unable to open file");
    for &datum in data.iter() {
        writeln!(file, "{}", datum).expect("Unable to write to file");
    }
}