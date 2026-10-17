//! Embeddable offline plate solver. No server, web UI, accounts or activation.
//! Start with [`Engine`]; lower-level algorithm modules remain available for research.

pub mod api;
pub mod attempt;
pub mod builder;
pub mod compat;
pub mod coordinates;
pub mod d80;
pub mod detect;
mod engine;
pub mod extract;
pub mod fingerprint;
pub mod geometry;
pub mod highpass;
pub mod hints;
pub mod image_io;
pub mod index;
pub mod learned;
pub mod lines;
pub mod measure;
pub mod patterns;
pub mod pipeline;
pub mod pixelgrid;
pub mod select;
pub mod skymask;
pub mod solve;
pub mod sphere;
pub mod types;
pub mod vet;
pub mod wcsfit;
pub use engine::{Engine, SolveOutput};
pub use extract::{ExtractOptions, extract};
pub use types::*;
