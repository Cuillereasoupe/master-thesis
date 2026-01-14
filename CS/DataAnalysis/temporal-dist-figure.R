library(tidyverse)
library(patchwork)

# Parse date and extract month
algae_data <- algae_data %>%
  mutate(
    observation_date = lubridate::dmy_hm(general_contexte_date_heure),
    month = factor(lubridate::month(observation_date, label = TRUE, abbr = TRUE),
                   levels = c("mai", "juin", "juil", "août", "sept", "oct"))
  )

# Left panel: Temporal distribution by month
temporal_plot <- algae_data %>%
  filter(!is.na(month)) %>%
  ggplot(aes(x = month)) +
  geom_bar(fill = "#F4A460", color = "white", alpha = 0.9) +
  labs(
    title = "Temporal distribution of observations",
    x = "Month",
    y = "Number of observations"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 12, face = "bold"),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 10)
  )

# Right panel: Pie chart for user type
user_summary <- algae_data %>%
  count(version) %>% 
  mutate(
    pct = n / sum(n) * 100,
    label = paste0(round(pct, 1), "%")
  )

pie_plot <- ggplot(user_summary, aes(x = "", y = n, fill = version)) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  geom_text(aes(label = label), 
            position = position_stack(vjust = 0.5), size = 4) +
  scale_fill_manual(
    values = c("pro" = "#4ECDC4", "science-participative" = "#F4A460"),
    labels = c("pro" = "Professional", "science-participative" = "Citizen science"),
    name = NULL
  ) +
  labs(title = "Observations by user type") +
  theme_void() +
  theme(
    plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
    legend.position = "bottom"
  )

# Combine
combined_plot <- temporal_plot + pie_plot +
  plot_layout(widths = c(1.5, 1))

# Save
ggsave("temporal_distribution.png", combined_plot, width = 9, height = 4, dpi = 300, bg = "white")

print(combined_plot)