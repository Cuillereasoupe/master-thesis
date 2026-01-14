library(tidyverse)
library(patchwork)

algae_data <- read_csv2("./obs_algues/obs_algues_corrected.csv") %>%
  janitor::clean_names()

# (a) Lake tour coverage
tour_data <- algae_data %>%
  filter(!is.na(observation_obs_pourcent_tour)) %>%
  count(observation_obs_pourcent_tour) %>%
  mutate(pct = n / sum(n) * 100)

p_a <- ggplot(tour_data, aes(x = observation_obs_pourcent_tour, y = pct)) +
  geom_col(fill = "#F4A460", color = "white", alpha = 0.9) +
  labs(title = "Lake tour coverage", x = NULL, y = "Percentage (%)") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# (b) User satisfaction ratings
# Adjust column name as needed
rating_data <- algae_data %>%
  filter(!is.na(feedback_retour_rating)) %>%
  count(feedback_retour_rating) %>%
  mutate(pct = n / sum(n) * 100)

mean_rating <- round(mean(algae_data$feedback_retour_rating, na.rm = TRUE), 2)

p_b <- ggplot(rating_data, aes(x = factor(feedback_retour_rating), y = pct)) +
  geom_col(fill = "#F4A460", color = "white", alpha = 0.9) +
  labs(title = paste0("User satisfaction (mean: ", mean_rating, "/5)"), x = "Rating (1-5)", y = "Percentage (%)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5))

# Combine side by side
combined <- p_a + p_b +
  plot_annotation(tag_levels = 'a', tag_prefix = '(', tag_suffix = ')')

# Save
ggsave("survey_feedback.png", combined, width = 9, height = 4, dpi = 300, bg = "white")

print(combined)