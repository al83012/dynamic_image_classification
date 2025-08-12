use std::collections::VecDeque;

use burn_train::{metric::MetricEntry, renderer::MetricState};

pub struct AvgMetric {
    entries: VecDeque<f64>,
    name: String,
    id: String,
    window_len: usize,
    started: bool,
}

impl AvgMetric {
    pub fn new(name: String, id: String, window_len: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(window_len),
            name,
            id,
            window_len,
            started: false,
        }
    }

    pub fn update(&mut self, next_value: f64) -> MetricState {
        if self.entries.len() >= self.window_len {
            self.entries.pop_front();
        }
        self.entries.push_back(next_value);

        let avg = self.current_avg();

        MetricState::Numeric(
            MetricEntry {
                name: self.name.clone(),
                formatted: self.name.clone(),
                serialize: self.id.clone(),
            },
            avg,
        )
    }
    pub fn current_avg(&self) -> f64 {
        self.entries.iter().fold(0.0, |acc, x| acc + *x) / self.entries.len() as f64
    }
    pub fn update_saturating(&mut self, next_value: f64) -> Option<MetricState> {
        if self.entries.len() >= self.window_len {
            self.entries.pop_front();
        }
        self.entries.push_back(next_value);

        let avg = self.current_avg();

        if self.entries.len() >= self.window_len {
            self.started = true;
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

    pub fn update_saturating_or_avg(&mut self, next_value: Option<f64>) -> Option<MetricState> {
        if let Some(next_val) = next_value {
            // If it is none, the sequence hasn't yet started
            self.update_saturating(next_val)
        } else if self.started {
            let avg = self.current_avg();
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
