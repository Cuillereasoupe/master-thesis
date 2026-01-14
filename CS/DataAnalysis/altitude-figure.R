library(tidyverse)

# Load your data (adjust path as needed)
algae_data <- read_csv2("./obs_algues/obs_algues_corrected.csv") %>%
  janitor::clean_names()

# Calculate mean for the dashed line
algae_data %>%
  summarise(
    mean_alt = mean(algae_data$general_contexte_loc_altitude, na.rm = TRUE),
    sd_alt = sd(algae_data$general_contexte_loc_altitude, na.rm = TRUE),
    n = sum(!is.na(algae_data$general_contexte_loc_altitude))
  )

# Clean altitude histogram
altitude_plot <- algae_data %>%
  filter(!is.na(algae_data$general_contexte_loc_altitude)) %>%
  ggplot(aes(x = algae_data$general_contexte_loc_altitude)) +
  geom_histogram(binwidth = 250, fill = "#4ECDC4", color = "white", alpha = 0.8) +
  geom_vline(xintercept = mean_alt, linetype = "dashed", color = "gray30", linewidth = 0.8) +
  annotate("text", x = mean_alt - 450, y = Inf, vjust = 2, hjust = 0,
           label = paste0("Mean: ", round(mean_alt), "m"), size = 3.5) +
  scale_x_continuous(breaks = seq(1000, 3000, 500), limits = c(1000, 3000)) +
  labs(
    title = "Altitude distribution of surveyed lakes",
    x = "Altitude (m)",
    y = "Number of observations"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 12, face = "bold"),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 10)
  )

# Save
ggsave("altitude_distribution.png", altitude_plot, width = 5, height = 4, dpi = 300, bg = "white")

print(altitude_plot)
