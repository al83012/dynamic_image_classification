use std::collections::VecDeque;

use burn_train::{metric::MetricEntry, renderer::MetricState};

pub struct AvgMetric {
    entries: VecDeque<f64>,
    name: String,
    id: String,
    window_len: usize,
}

impl AvgMetric {
    pub fn new(name: String, id: String, window_len: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(window_len),
            name,
            id,
            window_len,
        }
    }

    pub fn update(&mut self, next_value: f64) -> MetricState {
        if self.entries.len() >= self.window_len {
            self.entries.pop_front();
        }
        self.entries.push_back(next_value);

        let avg = self.entries.iter().fold(0.0, |acc, x| acc + *x) / self.entries.len() as f64;

        MetricState::Numeric(
            MetricEntry {
                name: self.name.clone(),
                formatted: self.name.clone(),
                serialize: self.id.clone(),
            },
            avg,
        )
    }
    pub fn update_saturating(&mut self, next_value: f64) -> Option<MetricState> {
        if self.entries.len() >= self.window_len {
            self.entries.pop_front();
        }
        self.entries.push_back(next_value);

        let avg = self.entries.iter().fold(0.0, |acc, x| acc + *x) / self.entries.len() as f64;

        if self.entries.len() >= self.window_len {
            Some(MetricState::Numeric(
                MetricEntry {
                    name: self.name.clone(),
                    formatted: self.name.clone(),
                    serialize: self.id.clone(),
                },
                avg,
            ))
        } else {
            None
        }
    }
}
