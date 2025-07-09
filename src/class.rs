use std::ops::{Deref, DerefMut};



#[derive(Clone, Copy, Debug)]
pub struct Classification{
    pub is_smoker: bool,
}


impl From<bool> for Classification {
    fn from(value: bool) -> Self {
        Self{is_smoker: value}
    }
}

impl Into<bool> for Classification {
    fn into(self) -> bool {
        self.is_smoker
    }
}

impl Deref for Classification {
    type Target = bool;
    fn deref(&self) -> &Self::Target {
       &self.is_smoker 
    }
}

impl DerefMut for Classification {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.is_smoker
    }
}
