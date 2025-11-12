use fltk::{
    app,
    draw::{self},
    enums::{Align, Color, Font},
    prelude::*,
    window,
};
use std::{
    cell::RefCell,
    rc::Rc,
    time::{SystemTime, UNIX_EPOCH},
};

const CHARACTERS: [char; 2] = ['0', '1'];
const FONT_SIZE: i32 = 16;
const TRAIL_DECAY: f32 = 0.05;
const DROP_RESET_PROBABILITY: f32 = 0.975;

#[derive(Clone)]
struct Cell {
    ch: char,
    intensity: f32,
}

struct MatrixState {
    columns: usize,
    rows: usize,
    drops: Vec<i32>,
    cells: Vec<Cell>,
    rng_state: u64,
}

impl MatrixState {
    fn new(width: i32, height: i32) -> Self {
        let mut state = MatrixState {
            columns: 0,
            rows: 0,
            drops: Vec::new(),
            cells: Vec::new(),
            rng_state: Self::seed(),
        };
        state.resize(width, height);
        state
    }

    fn seed() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x1234_5678_9abc_def0)
    }

    fn resize(&mut self, width: i32, height: i32) {
        let col_count = ((width.max(1) as usize) + FONT_SIZE as usize - 1) / FONT_SIZE as usize;
        let row_count = ((height.max(1) as usize) + FONT_SIZE as usize - 1) / FONT_SIZE as usize;

        self.columns = col_count.max(1);
        self.rows = row_count.max(1);

        self.cells = vec![
            Cell {
                ch: '0',
                intensity: 0.0,
            };
            self.columns * self.rows
        ];

        self.drops = (0..self.columns)
            .map(|_| {
                let offset = (self.random_f32() * self.rows as f32) as i32;
                -offset
            })
            .collect();
    }

    fn tick(&mut self) {
        for cell in &mut self.cells {
            cell.intensity *= 1.0 - TRAIL_DECAY;
            if cell.intensity < 0.01 {
                cell.intensity = 0.0;
            }
        }

        let rows = self.rows;
        let columns = self.columns;

        // Collect random values first to avoid borrow conflicts
        let mut random_chars = Vec::new();
        let mut random_resets = Vec::new();
        let mut random_offsets = Vec::new();

        for _ in 0..self.drops.len() {
            random_chars.push(self.random_char());
            random_resets.push(self.random_f32());
            random_offsets.push((self.random_f32() * rows as f32) as i32);
        }

        // Now update drops
        for (col, drop) in self.drops.iter_mut().enumerate() {
            if *drop < 0 {
                *drop += 1;
                continue;
            }

            let row = *drop as usize;
            if row < rows {
                let idx = row * columns + col;
                if idx < self.cells.len() {
                    self.cells[idx].ch = random_chars[col];
                    self.cells[idx].intensity = 1.0;
                }
            }

            *drop += 1;

            if *drop as usize >= rows
                && random_resets.get(col).copied().unwrap_or(0.0) > DROP_RESET_PROBABILITY
            {
                let offset = random_offsets.get(col).copied().unwrap_or(0);
                *drop = -offset;
            }
        }
    }

    fn index(&self, col: usize, row: usize) -> usize {
        row * self.columns + col
    }

    fn random_char(&mut self) -> char {
        let idx = (self.next_rand() % CHARACTERS.len() as u64) as usize;
        CHARACTERS[idx]
    }

    fn random_f32(&mut self) -> f32 {
        const SCALE: f64 = 1.0 / (u64::MAX as f64);
        (self.next_rand() as f64 * SCALE) as f32
    }

    fn next_rand(&mut self) -> u64 {
        if self.rng_state == 0 {
            self.rng_state = 0x1234_5678_9abc_def0;
        }
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.rng_state
    }
}

pub fn install_matrix_background(wind: &mut window::Window) {
    let state = Rc::new(RefCell::new(MatrixState::new(wind.width(), wind.height())));

    // Set the window's draw callback to render the matrix background
    wind.draw({
        let state = state.clone();
        move |w| {
            let matrix = state.borrow();

            // Draw black background
            draw::set_draw_color(Color::Black);
            draw::draw_rectf(w.x(), w.y(), w.width(), w.height());

            draw::set_font(Font::Courier, FONT_SIZE);

            let mut buf = [0u8; 4];

            for row in 0..matrix.rows {
                for col in 0..matrix.columns {
                    let cell = &matrix.cells[matrix.index(col, row)];
                    if cell.intensity <= 0.0 {
                        continue;
                    }

                    let brightness = (cell.intensity.clamp(0.0, 1.0) * 255.0) as u8;
                    let color = if cell.intensity > 0.9 {
                        Color::from_rgb(180, 255, 200)
                    } else {
                        Color::from_rgb(0, brightness.max(20), 0)
                    };

                    draw::set_draw_color(color);
                    let text = cell.ch.encode_utf8(&mut buf);
                    let x = w.x() + (col as i32) * FONT_SIZE;
                    let y = w.y() + ((row + 1) as i32) * FONT_SIZE;
                    draw::draw_text2(
                        text,
                        x,
                        y - FONT_SIZE,
                        FONT_SIZE,
                        FONT_SIZE,
                        Align::Inside | Align::Left,
                    );
                }
            }
        }
    });

    // Handle window resize
    {
        let state = state.clone();
        let mut wind_clone = wind.clone();
        wind.resize_callback(move |_, _, _, w, h| {
            {
                let mut matrix = state.borrow_mut();
                matrix.resize(w, h);
            }
            wind_clone.redraw();
        });
    }

    // Animate the matrix background
    {
        let state = state.clone();
        let mut wind_clone = wind.clone();
        app::add_timeout3(0.033, move |handle| {
            {
                let mut matrix = state.borrow_mut();
                matrix.tick();
            }
            wind_clone.redraw();
            app::repeat_timeout3(0.033, handle);
        });
    }
}
