"""
Configuration module for DataGen
"""
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any


@dataclass
class SamplingConfig:
    """Configuration for text generation sampling parameters"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 1024


@dataclass
class QualityConfig:
    """Configuration for quality filtering"""
    enable_filtering: bool = True
    min_length: int = 10
    max_length: int = 2048
    min_instruction_length: int = 3
    min_response_length: int = 5
    perplexity_threshold: Optional[float] = None
    similarity_threshold: float = 0.85  # For duplicate detection
    custom_filters: List[Any] = field(default_factory=list)


@dataclass
class PrivacyConfig:
    """Configuration for privacy preservation"""
    enable_privacy: bool = False
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    enable_content_filtering: bool = True
    sensitive_terms: List[str] = field(default_factory=list)


@dataclass
class GenerationConfig:
    """Configuration for generation engine"""
    model_name: str = "gpt-3.5-turbo"
    backend: str = "openai"
    self_instruct: bool = True
    evol_instruct: bool = False
    evol_rounds: int = 1
    prompt_templates_dir: Optional[str] = None
    prompt_template: Optional[str] = None
    custom_prompt_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    """Configuration for dataset metrics and evaluation."""
    
    # Computation flags
    compute_perplexity: bool = True
    compute_topics: bool = True
    create_visualizations: bool = True
    
    # Visualization settings
    visualization_format: str = "png"
    visualization_dpi: int = 300
    plots_cmap: str = "viridis"
    
    # Model settings
    perplexity_model: str = "distilgpt2"
    perplexity_batch_size: int = 8
    perplexity_max_length: int = 1024
    
    # Topic extraction settings
    topic_method: str = "auto"  # "auto", "lda", "bertopic", or "frequency"
    num_topics: int = 10
    top_n_words: int = 10
    
    # Diversity settings
    sample_size: int = 100  # For similarity calculations
    ngram_max: int = 4  # Maximum n-gram size for uniqueness
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class RLTuningConfig:
    """Configuration for reinforcement learning-guided data generation."""
    
    enable_rl_tuning: bool = False
    num_iterations: int = 10
    batch_size: int = 100
    rl_algorithm: str = "random_search"  # Options: "random_search", "reinforce"
    reward_metric: str = "accuracy"
    target_model_path: Optional[str] = None
    validation_data_path: Optional[str] = None
    
    # Parameter range configuration
    max_temperature: float = 1.0
    min_temperature: float = 0.1
    max_top_p: float = 1.0
    min_top_p: float = 0.1
    temperature_step: float = 0.05
    top_p_step: float = 0.05
    generation_methods: List[str] = field(default_factory=lambda: ["self_instruct", "evol_instruct"])
    
    # REINFORCE (policy gradient) specific parameters
    learning_rate: float = 0.001
    policy_hidden_dim: int = 64  # Hidden layer size for policy network
    gamma: float = 0.99  # Discount factor for future rewards
    normalize_rewards: bool = True  # Whether to normalize rewards
    entropy_coef: float = 0.01  # Entropy coefficient for exploration


@dataclass
class Config:
    """Main configuration for DataGen"""
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    rl_tuning: RLTuningConfig = field(default_factory=RLTuningConfig)
    output_format: str = "jsonl"
    api_keys: Dict[str, str] = field(default_factory=dict)
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from a YAML file"""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        config = cls()
        
        if "sampling" in config_dict:
            config.sampling = SamplingConfig(**config_dict["sampling"])
            
        if "quality" in config_dict:
            config.quality = QualityConfig(**config_dict["quality"])
            
        if "privacy" in config_dict:
            config.privacy = PrivacyConfig(**config_dict["privacy"])
            
        if "generation" in config_dict:
            config.generation = GenerationConfig(**config_dict["generation"])
            
        if "metrics" in config_dict:
            config.metrics = MetricsConfig(**config_dict["metrics"])
            
        if "rl_tuning" in config_dict:
            config.rl_tuning = RLTuningConfig(**config_dict["rl_tuning"])
            
        for key, value in config_dict.items():
            if key not in ["sampling", "quality", "privacy", "generation", "metrics", "rl_tuning"]:
                setattr(config, key, value)
                
        return config
    
    def save(self, config_path: str) -> None:
        """Save configuration to a YAML file"""
        config_dict = {
            "sampling": {k: v for k, v in self.sampling.__dict__.items()},
            "quality": {k: v for k, v in self.quality.__dict__.items()},
            "privacy": {k: v for k, v in self.privacy.__dict__.items()},
            "generation": {k: v for k, v in self.generation.__dict__.items()},
            "metrics": {k: v for k, v in self.metrics.__dict__.items()},
            "rl_tuning": {k: v for k, v in self.rl_tuning.__dict__.items()},
            "output_format": self.output_format,
            "api_keys": self.api_keys,
            "log_level": self.log_level,
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def get_preset(self, preset_name: str) -> "Config":
        """Load a preset configuration"""
        presets = {
            "instruction_tuning": self._preset_instruction_tuning(),
            "domain_adaptation": self._preset_domain_adaptation(),
            "privacy_focused": self._preset_privacy_focused(),
            "quality_focused": self._preset_quality_focused(),
        }
        
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found. Available presets: {list(presets.keys())}")
            
        return presets[preset_name]
    
    def _preset_instruction_tuning(self) -> "Config":
        """Preset for instruction tuning"""
        config = Config()
        config.generation.self_instruct = True
        config.generation.evol_instruct = False
        config.sampling.temperature = 0.7
        config.quality.enable_filtering = True
        return config
    
    def _preset_domain_adaptation(self) -> "Config":
        """Preset for domain adaptation"""
        config = Config()
        config.generation.self_instruct = True
        config.sampling.temperature = 0.8
        config.sampling.top_p = 0.95
        return config
    
    def _preset_privacy_focused(self) -> "Config":
        """Preset with strong privacy constraints"""
        config = Config()
        config.privacy.enable_privacy = True
        config.privacy.differential_privacy = True
        config.privacy.dp_epsilon = 0.5
        config.privacy.enable_content_filtering = True
        return config
    
    def _preset_quality_focused(self) -> "Config":
        """Preset focused on generation quality"""
        config = Config()
        config.quality.enable_filtering = True
        config.quality.perplexity_threshold = 50.0
        config.quality.similarity_threshold = 0.9
        config.sampling.temperature = 0.6
        return config 