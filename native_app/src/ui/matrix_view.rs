
use fltk::{
    draw,
    enums::{Align, Color},
    prelude::*,
    table,
};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct MatrixView {
    pub table: table::Table,
    data: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl MatrixView {
    pub fn new() -> Self {
        let mut table = table::Table::default().with_size(800, 800);
        table.set_rows(16);
        table.set_cols(16);
        table.set_col_header(true);
        table.set_col_resize(true);
        table.set_row_header(true);
        table.set_row_resize(true);
        table.end();

        let data = Arc::new(Mutex::new(vec![vec![0.0; 16]; 16]));

        let mut s = Self { table, data };

        s.table.draw_cell(move |_, context, _row, _col, x, y, w, h| {
            if context == table::TableContext::Cell {
                // TODO: Implement heatmap drawing logic
                draw::draw_rect_fill(x, y, w, h, Color::from_hex(0x0a0e27));
                draw::set_draw_color(Color::Green);
                draw::draw_text2("0", x, y, w, h, Align::Center);
            }
        });

        s
    }
}