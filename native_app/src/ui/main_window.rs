// ... existing code ...
use super::sidebar::Sidebar;
use super::status_bar::StatusBar;
use super::chat_panel::ChatPanel;

pub struct MainWindow {
// ... existing code ...
    pub sidebar: Sidebar,
    pub status_bar: StatusBar,
    pub chat_panel: ChatPanel,
}

impl MainWindow {
// ... existing code ...
        let main_flex = Flex::default_fill().row();

        let sidebar = Sidebar::new();

        let content_flex = Flex::default().column();
        let tabs = Tabs::default();
// ... existing code ...
        content_flex.end();

        let chat_panel = ChatPanel::new();

        main_flex.end();
        wind.end();
        wind.make_resizable(true);
// ... existing code ...
            sidebar,
            status_bar,
            chat_panel,
        }
    }
}
