use crate::dungeons::{read_recent_deliberations, DungeonConfig, DungeonManager};
use crate::models::AppState;
use fltk::enums::Color;
use fltk::{prelude::*, *};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct DungeonsView {
    pub start_btn: button::Button,
    pub stop_btn: button::Button,
    pub log_display: text::TextDisplay,
    pub status_display: text::TextDisplay,
    pub copy_status_btn: button::Button,
    pub copy_log_btn: button::Button,
}

pub fn create_dungeons_tab(tabs: &mut group::Tabs, state: Arc<Mutex<AppState>>) -> DungeonsView {
    let mut group = group::Group::default().with_label("🗺️ Dungeons");
    group.begin();
    let root = group::Flex::default().with_size(1450, 950).column();

    let mut control_row = group::Flex::default().with_size(1450, 80).row();
    let mut start_btn = button::Button::default()
        .with_size(160, 80)
        .with_label("▶ Start");
    start_btn.set_color(Color::from_hex(0x00ff64));
    start_btn.set_label_color(Color::Black);

    let mut stop_btn = button::Button::default()
        .with_size(160, 80)
        .with_label("⏹ Stop");
    stop_btn.set_color(Color::from_hex(0xff6464));
    stop_btn.set_label_color(Color::White);

    control_row.add(&start_btn);
    control_row.add(&stop_btn);
    control_row.end();

    let status_buffer = text::TextBuffer::default();
    let mut status_display = text::TextDisplay::default().with_size(1450, 100);
    status_display.set_buffer(status_buffer.clone());
    status_display
        .buffer()
        .unwrap()
        .set_text("Select a dungeon and press start to begin the ritual.");
    status_display.set_text_color(Color::from_hex(0x00d4ff));

    let log_buffer = text::TextBuffer::default();
    let mut log_display = text::TextDisplay::default().with_size(1450, 720);
    log_display.set_buffer(log_buffer.clone());
    log_display.set_text_color(Color::White);

    let mut copy_row = group::Flex::default().with_size(1450, 50).row();
    let mut copy_status_btn = button::Button::default()
        .with_size(120, 50)
        .with_label("Copy Status");
    copy_status_btn.set_color(Color::from_hex(0x1a1f3a));
    copy_status_btn.set_label_color(Color::from_hex(0x00d4ff));
    copy_row.add(&copy_status_btn);

    let mut copy_log_btn = button::Button::default()
        .with_size(120, 50)
        .with_label("Copy Log");
    copy_log_btn.set_color(Color::from_hex(0x1a1f3a));
    copy_log_btn.set_label_color(Color::from_hex(0x00d4ff));
    copy_row.add(&copy_log_btn);
    copy_row.end();

    root.end();
    group.end();
    tabs.add(&group);

    let dungeon_id = "trolley_temple_001".to_string();
    let cfg = DungeonConfig::new(&dungeon_id);

    {
        let cfg_clone = cfg.clone();
        let log_display = log_display.clone();
        app::add_timeout3(0.5, move |handle| {
            let text = read_recent_deliberations(&cfg_clone, 32);
            log_display.buffer().unwrap().set_text(&text);
            app::repeat_timeout3(0.5, handle);
        });
    }

    start_btn.set_callback({
        let state = state.clone();
        let cfg = cfg.clone();
        let status = status_display.clone();
        move |_| {
            let mgr = DungeonManager::new(state.clone(), &cfg.id);
            mgr.start_simulated_run();
            status
                .buffer()
                .unwrap()
                .set_text("Dungeon run initiated. Watching deliberations...");
        }
    });

    stop_btn.set_callback({
        let state = state.clone();
        let cfg = cfg.clone();
        let status = status_display.clone();
        move |_| {
            let mgr = DungeonManager::new(state.clone(), &cfg.id);
            mgr.stop();
            status.buffer().unwrap().set_text("Dungeon run stopped.");
        }
    });

    DungeonsView {
        start_btn,
        stop_btn,
        log_display,
        status_display,
        copy_status_btn,
        copy_log_btn,
    }
}
