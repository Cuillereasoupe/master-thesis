library(tidyverse)
library(patchwork)

algae_data <- read_csv2("./obs_algues/obs_algues_corrected.csv") %>%
  janitor::clean_names()

# (a) Algae behavior (only when algae present)
behavior_data <- algae_data %>%
  filter(observation_obs_bloom == "yes", !is.na(observation_algae_alg_compor)) %>%
  count(observation_algae_alg_compor) %>%
  mutate(pct = n / sum(n) * 100)

p_a <- ggplot(behavior_data, aes(x = observation_algae_alg_compor, y = pct)) +
  geom_col(fill = "#F4A460", color = "white", alpha = 0.9) +
  labs(title = "Algae behavior (when present)", x = NULL, y = "Percentage (%)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5))

# (b) Algae color (when present)
color_data <- algae_data %>%
  filter(observation_obs_bloom == "yes", !is.na(observation_algae_alg_couleur)) %>%
  count(observation_algae_alg_couleur) %>%
  mutate(pct = n / sum(n) * 100)

p_b <- ggplot(color_data, aes(x = reorder(observation_algae_alg_couleur, -pct), y = pct)) +
  geom_col(fill = "#F4A460", color = "white", alpha = 0.9) +
  labs(title = "Algae color (when present)", x = NULL, y = "Percentage (%)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5))

# (c) Substrate type - count each type properly
substrate_counts <- algae_data %>%
  filter(observation_obs_bloom == "yes", !is.na(observation_algae_alg_substrat)) %>%
  summarise(
    rocks = sum(str_detect(observation_algae_alg_substrat, "rocks")),
    `clay-sand` = sum(str_detect(observation_algae_alg_substrat, "clay-sand")),
    `sand-gravel` = sum(str_detect(observation_algae_alg_substrat, "sand-gravel")),
    pebbles = sum(str_detect(observation_algae_alg_substrat, "pebbles")),
    other = sum(str_detect(observation_algae_alg_substrat, "other")),
    unknown = sum(str_detect(observation_algae_alg_substrat, "unknown"))
  ) %>%
  pivot_longer(everything(), names_to = "substrate_type", values_to = "n") %>%
  mutate(pct = n / sum(n) * 100)

print(substrate_counts)

p_c <- ggplot(substrate_counts, aes(x = reorder(substrate_type, -pct), y = pct)) +
  geom_col(fill = "#4ECDC4", color = "white", alpha = 0.9) +
  labs(title = "Substrate type", x = NULL, y = "Percentage (%)") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# (d) Shore coverage percentage (when algae present)
coverage_data <- algae_data %>%
  filter(observation_obs_bloom == "yes", !is.na(observation_algae_alg_recouvr_pourcent))

mean_coverage <- round(mean(coverage_data$observation_algae_alg_recouvr_pourcent, na.rm = TRUE), 1)

p_d <- coverage_data %>%
  count(observation_algae_alg_recouvr_pourcent) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = factor(observation_algae_alg_recouvr_pourcent), y = pct)) +
  geom_col(fill = "#F4A460", color = "white", alpha = 0.9) +
  geom_vline(xintercept = mean_coverage / 10 + 1, linetype = "dashed", color = "gray30", linewidth = 0.8) +
  annotate("text", x = 7, y = Inf, vjust = 2, hjust = 0,
           label = paste0("Mean: ", mean_coverage, "%"), size = 3.5) +
  labs(title = "Shore coverage percentage (when algae present)", x = "Coverage (%)", y = "Percentage (%)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5))

# Combine with panel labels
combined <- (p_a + p_b) / (p_c + p_d) +
  plot_annotation(tag_levels = 'a', tag_prefix = '(', tag_suffix = ')')

# Save
ggsave("algae_characteristics.png", combined, width = 9, height = 8, dpi = 300, bg = "white")

print(combined)