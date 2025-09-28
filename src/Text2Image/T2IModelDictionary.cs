using System.Collections.Generic;
using System.Collections.Frozen;

namespace SwarmUI.Text2Image;

public enum ModelType
{
    TextToImage,
    TextToVideo,
    ImageToVideo,
    TextAndImageToVideo,
    ImageEdit,
}

public enum ModelArchitecture
{
    UNet,
    Dit,
    Cascade,
    MLLM,
}

public enum PredictionType
{
    Epsilon,
    VPrediction,
    RectifiedFlow,
}

/// <summary>Holds detailed information about a specific model class.</summary>
public record class ModelInfo(
    string BaseModel,
    ModelType Type,
    ModelArchitecture Architecture,
    PredictionType PredType,
    int DefaultWidth,
    int DefaultHeight,
    List<string> TextEncoders = null,
    string LatentNode = "EmptyLatentImage",
    string DefaultSampler = "euler",
    string DefaultScheduler = "simple",
    int DefaultFrames = 0,
    string DefaultVAE = null,
    double? DefaultSigmaShift = null
);

/// <summary>A centralized dictionary of all model-specific parameters and information.</summary>
public static class ModelDictionary
{
    /// <summary>The main dictionary mapping a model's compatibility class to its detailed information.</summary>
    public static FrozenDictionary<string, ModelInfo> Models;

    /// <summary>Register a new model type.</summary>
    public static void Register(Dictionary<string, ModelInfo> models, ModelInfo info)
    {
        models.Add(info.BaseModel, info);
    }

    /// <summary>(Called during startup) registers all default model types.</summary>
    public static void RegisterDefaults()
    {
        var models = new Dictionary<string, ModelInfo>();

        // ================= UNet Models =================
        Register(models, new("stable-diffusion-v1", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, 512, 512, ["clip_l"]));
        Register(models, new("stable-diffusion-v2", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.VPrediction, 768, 768, ["clip_g"]));
        Register(models, new("stable-diffusion-xl-v1", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, 1024, 1024, ["clip_l", "clip_g"], DefaultVAE: "sdxl-vae"));
        Register(models, new("segmind-stable-diffusion-1b", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, 1024, 1024, ["clip_l", "clip_g"], DefaultVAE: "sdxl-vae"));
        Register(models, new("stable-video-diffusion-img2vid-v1", ModelType.ImageToVideo, ModelArchitecture.UNet, PredictionType.Epsilon, 1024, 576, DefaultFrames: 25));

        // ================= DiT Models =================
        Register(models, new("stable-diffusion-v3-medium", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, 1024, 1024, ["clip_l", "clip_g", "t5xxl"], "EmptySD3LatentImage", DefaultVAE: "sd35-vae", DefaultSigmaShift: 3.0));
        Register(models, new("stable-diffusion-v3.5-large", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, 1024, 1024, ["clip_l", "clip_g", "t5xxl"], "EmptySD3LatentImage", DefaultVAE: "sd35-vae", DefaultSigmaShift: 3.0));
        Register(models, new("flux-1", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, 1024, 1024, ["clip_l", "t5xxl"], "EmptySD3LatentImage", DefaultScheduler: "simple", DefaultVAE: "flux-ae"));
        Register(models, new("pixart-ms-sigma-xl-2", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, 1024, 1024, ["t5xxl"], DefaultVAE: "sdxl-vae"));
        Register(models, new("nvidia-sana-1600", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, 1024, 1024, LatentNode: "EmptySanaLatentImage", DefaultVAE: "sana-dcae"));
        Register(models, new("auraflow-v1", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, 1024, 1024, ["clip_l", "clip_g", "t5xxl"], DefaultVAE: "flux-ae", DefaultSigmaShift: 1.73));
        Register(models, new("lumina-2", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, 1024, 1024, ["gemma2"], DefaultVAE: "flux-ae", DefaultSigmaShift: 6.0));
        Register(models, new("hidream-i1", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, 1024, 1024, ["long_clip_l", "long_clip_g", "t5xxl", "llama3.1_8b"], "EmptySD3LatentImage", DefaultVAE: "flux-ae", DefaultSigmaShift: 3.0));
        Register(models, new("nvidia-cosmos-predict2-t2i-14b", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, 1024, 1024, DefaultVAE: "wan21-vae"));
        Register(models, new("qwen-image", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, 1328, 1328, ["qwen-2.5-vl"], "EmptySD3LatentImage", DefaultScheduler: "simple", DefaultVAE: "qwen-image-vae", DefaultSigmaShift: 3.0));
        Register(models, new("hunyuan-image-2_1", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, 2048, 2048, ["qwen-2.5-vl", "byt5-small"], "EmptyHunyuanImageLatent", DefaultVAE: "hunyuan-image-2_1-vae", DefaultSigmaShift: 3.0));
        Register(models, new("hunyuan-video", ModelType.TextToVideo, ModelArchitecture.Dit, PredictionType.RectifiedFlow, 720, 720, ["clip_l", "llava"], "EmptyHunyuanLatentVideo", DefaultFrames: 73, DefaultVAE: "hunyuan-video-vae", DefaultSigmaShift: 3.0));
        Register(models, new("wan-21", ModelType.TextAndImageToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, 960, 960, ["umt5xxl"], "EmptyHunyuanLatentVideo", DefaultScheduler: "simple", DefaultFrames: 81, DefaultVAE: "wan21-vae", DefaultSigmaShift: 3.0));
        Register(models, new("wan-22", ModelType.TextAndImageToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, 960, 960, ["umt5xxl"], "Wan22ImageToVideoLatent", DefaultScheduler: "simple", DefaultFrames: 81, DefaultVAE: "wan22-vae", DefaultSigmaShift: 3.0));
        Register(models, new("nvidia-cosmos-1", ModelType.TextAndImageToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, 960, 960, ["old_t5xxl"], "EmptyCosmosLatentVideo", "res_multistep", "karras", 121, "cosmos-vae"));
        Register(models, new("chroma", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, 1024, 1024, ["clip_l", "t5xxl"], "EmptySD3LatentImage", DefaultVAE: "flux-ae", DefaultSigmaShift: 3.0));
        Register(models, new("lightricks-ltx-video", ModelType.TextToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, 768, 512, ["t5xxl"], "EmptyLTXVLatentVideo", DefaultScheduler: "ltxv", DefaultFrames: 97, DefaultVAE: "ltxv-vae"));
        Register(models, new("genmo-mochi-1", ModelType.TextToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, 848, 480, ["t5xxl"], "EmptyMochiLatentVideo", DefaultFrames: 25, DefaultVAE: "mochi-vae"));

        // ================= Cascade Models =================
        Register(models, new("stable-cascade-v1", ModelType.TextToImage, ModelArchitecture.Cascade, PredictionType.RectifiedFlow, 1024, 1024, LatentNode: "StableCascade_EmptyLatentImage", DefaultSampler: "euler_ancestral", DefaultScheduler: "simple"));

        // ================= MLLM Models =================
        Register(models, new("omnigen-2", ModelType.TextToImage, ModelArchitecture.MLLM, PredictionType.Epsilon, 1024, 1024, ["qwen-2.5-vl"], "EmptySD3LatentImage", DefaultScheduler: "simple", DefaultVAE: "flux-ae"));

        Models = models.ToFrozenDictionary();
    }

    /// <summary>Gets the ModelInfo for a given compatibility class, or null if not found.</summary>
    public static ModelInfo GetModel(string baseModel) =>
        !string.IsNullOrEmpty(baseModel) && Models is not null && Models.TryGetValue(baseModel, out var modelInfo) ? modelInfo : null;
}
