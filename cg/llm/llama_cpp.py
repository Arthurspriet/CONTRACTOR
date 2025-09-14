"""
Local LLM integration using llama.cpp with CUDA support
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import llama_cpp
except ImportError:
    raise ImportError(
        "llama-cpp-python is not installed. "
        "Install with: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python"
    )

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LocalLlama:
    """
    Local LLM wrapper using llama.cpp with CUDA acceleration
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        n_threads: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        verbose: bool = False
    ):
        """
        Initialize LocalLlama with CUDA-enabled llama.cpp
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context size (default from env)
            n_gpu_layers: Number of layers to offload to GPU (default from env)
            n_threads: Number of CPU threads (default from env)
            temperature: Sampling temperature (default from env)
            top_p: Top-p sampling (default from env)
            verbose: Enable verbose logging
        """
        self.model_path = model_path or os.getenv("LLM_MODEL_PATH", "models/llm/mistral-7b-instruct.Q5_K_M.gguf")
        self.n_ctx = n_ctx or int(os.getenv("LLM_CTX_SIZE", "8192"))
        self.n_gpu_layers = n_gpu_layers or int(os.getenv("LLM_N_GPU_LAYERS", "999"))
        self.n_threads = n_threads or int(os.getenv("LLM_N_THREADS", "8"))
        self.temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.top_p = top_p or float(os.getenv("LLM_TOP_P", "0.9"))
        self.verbose = verbose or os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
        
        self._llm = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the GGUF model with CUDA support"""
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please download a GGUF model and place it at the specified path.\n"
                f"Recommended: Mistral-7B-Instruct or Qwen2-7B-Instruct (Q5_K_M quantization)"
            )
        
        if self.verbose:
            print(f"Loading model: {model_path}")
            print(f"Context size: {self.n_ctx}")
            print(f"GPU layers: {self.n_gpu_layers}")
            print(f"CPU threads: {self.n_threads}")
        
        try:
            self._llm = llama_cpp.Llama(
                model_path=str(model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose,
                # Enable CUDA optimizations
                use_mmap=True,
                use_mlock=False,
                # Ensure deterministic results for testing
                seed=42 if os.getenv("USE_MOCK_LLM", "false").lower() != "true" else -1,
            )
            
            if self.verbose:
                print("✓ Model loaded successfully with CUDA acceleration")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list] = None
    ) -> str:
        """
        Generate text using the local LLM
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (overrides instance default)
            top_p: Top-p sampling (overrides instance default)
            stop: List of stop sequences
            
        Returns:
            Generated text string
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Use instance defaults if not specified
        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p
        
        try:
            # Generate response
            response = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=tp,
                stop=stop or [],
                echo=False,  # Don't echo the prompt
                stream=False
            )
            
            # Extract generated text
            generated_text = response.get("choices", [{}])[0].get("text", "").strip()
            
            if self.verbose:
                print(f"Generated {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "model_loaded": self._llm is not None
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_llm') and self._llm is not None:
            del self._llm


# Mock LLM for testing
class MockLocalLlama(LocalLlama):
    """Mock version of LocalLlama for testing without actual model files"""
    
    def __init__(self, **kwargs):
        # Skip the parent __init__ to avoid loading actual model
        self.model_path = kwargs.get("model_path", "mock://model.gguf")
        self.n_ctx = kwargs.get("n_ctx", 8192)
        self.n_gpu_layers = kwargs.get("n_gpu_layers", 999)
        self.n_threads = kwargs.get("n_threads", 8)
        self.temperature = kwargs.get("temperature", 0.1)
        self.top_p = kwargs.get("top_p", 0.9)
        self.verbose = kwargs.get("verbose", False)
        self._llm = "mock"  # Indicate model is "loaded"
    
    def _load_model(self) -> None:
        """Mock model loading"""
        if self.verbose:
            print("✓ Mock model loaded successfully")
    
    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        """Generate mock responses for testing"""
        # Simple mock responses based on prompt content
        prompt_lower = prompt.lower()
        
        if "hello" in prompt_lower:
            return "Hello! I'm a mock LLM response."
        elif "clause" in prompt_lower:
            return "This clause should be revised to include standard liability limitations."
        elif "email" in prompt_lower:
            return "Dear counterpart, I'd like to discuss the following contract terms..."
        elif "risk" in prompt_lower:
            return "The risk level for this clause is amber due to potential liability exposure."
        else:
            return f"Mock response to: {prompt[:50]}..." 