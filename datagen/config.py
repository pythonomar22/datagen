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
    custom_filters: List[str] = field(default_factory=list)


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
    custom_prompt_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration for DataGen"""
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
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
            
        for key, value in config_dict.items():
            if key not in ["sampling", "quality", "privacy", "generation"]:
                setattr(config, key, value)
                
        return config
    
    def save(self, config_path: str) -> None:
        """Save configuration to a YAML file"""
        config_dict = {
            "sampling": {k: v for k, v in self.sampling.__dict__.items()},
            "quality": {k: v for k, v in self.quality.__dict__.items()},
            "privacy": {k: v for k, v in self.privacy.__dict__.items()},
            "generation": {k: v for k, v in self.generation.__dict__.items()},
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