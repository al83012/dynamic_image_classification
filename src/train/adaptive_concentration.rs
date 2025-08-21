const ITERATION_GOAL: f32 = 8.0;

pub struct AdaptiveConcentrationGoal {
    pub current_goal: f32,
}

impl AdaptiveConcentrationGoal {
    pub fn update_goal(&mut self, avg_iter_count: f32) {
        if (avg_iter_count - ITERATION_GOAL).abs() < 0.4 {
            return;
        }
        if avg_iter_count > ITERATION_GOAL {
            self.current_goal -= 0.01;
        }
        if avg_iter_count < ITERATION_GOAL {
            self.current_goal += 0.01;
        }
    }
    pub fn new(starting_goal: f32) -> Self {
        Self {
            current_goal: starting_goal,
        }
    }
}

impl Default for AdaptiveConcentrationGoal {
    fn default() -> Self {
        Self { current_goal: 0.4 }
    }
}
