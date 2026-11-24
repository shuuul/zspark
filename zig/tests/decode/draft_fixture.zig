//! What the draft tests of both models share: which positions an oracle drafter gets wrong. No device.

pub fn never(_: usize) bool {
    return false;
}
pub fn always(_: usize) bool {
    return true;
}
pub fn everyThird(position: usize) bool {
    return position % 3 == 0;
}
pub fn everyFifth(position: usize) bool {
    return position % 5 == 1;
}
