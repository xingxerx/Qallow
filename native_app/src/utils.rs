use chrono::{DateTime, Utc};

pub fn format_timestamp(dt: DateTime<Utc>) -> String {
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}

pub fn format_uptime(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    format!("{}h {}m {}s", hours, minutes, secs)
}

pub fn format_memory(gb: f64) -> String {
    format!("{:.1} GB", gb)
}

pub fn format_percentage(value: f64) -> String {
    format!("{:.2}%", value * 100.0)
}

pub fn get_status_color(value: f64) -> &'static str {
    match value {
        v if v >= 0.95 => "🟢", // Green - Excellent
        v if v >= 0.85 => "🟡", // Yellow - Good
        v if v >= 0.70 => "🟠", // Orange - Fair
        _ => "🔴",              // Red - Poor
    }
}

pub fn get_status_text(value: f64) -> &'static str {
    match value {
        v if v >= 0.95 => "Excellent",
        v if v >= 0.85 => "Good",
        v if v >= 0.70 => "Fair",
        _ => "Poor",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_uptime() {
        assert_eq!(format_uptime(3661), "1h 1m 1s");
        assert_eq!(format_uptime(60), "0h 1m 0s");
        assert_eq!(format_uptime(3600), "1h 0m 0s");
    }

    #[test]
    fn test_format_memory() {
        assert_eq!(format_memory(8.5), "8.5 GB");
        assert_eq!(format_memory(4.2), "4.2 GB");
    }

    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(0.95), "95.00%");
        assert_eq!(format_percentage(0.5), "50.00%");
    }

    #[test]
    fn test_get_status_color() {
        assert_eq!(get_status_color(0.99), "🟢");
        assert_eq!(get_status_color(0.90), "🟡");
        assert_eq!(get_status_color(0.75), "🟠");
        assert_eq!(get_status_color(0.50), "🔴");
    }
}

