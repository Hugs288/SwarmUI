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
    List<string> TextEncoders,
    string VAE,
    int DefaultWidth,
    int DefaultHeight,
    double DefaultCFG,
    int DefaultSteps,
    string LatentNode = "EmptyLatentImage",
    string DefaultSampler = "euler",
    string DefaultScheduler = "simple",
    int DefaultFrames = 0,
    double? DefaultSigmaShift = null,
    string SigmaShiftNode = null,
    string UniqueClipLoader = null,
    string UniqueModelLoader = null,
    string ClipType = null
);

/// <summary>A centralized dictionary of all model-specific parameters and information.</summary>
public static class ModelDictionary
{
    /// <summary>The main dictionary mapping a model's compatibility class to its detailed information.</summary>
    public static FrozenDictionary<string, ModelInfo> Models;

    /// <summary>Static constructor to initialize the model dictionary.</summary>
    static ModelDictionary()
    {
        RegisterDefaults();
    }

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
        Register(models, new("stable-diffusion-v1", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, ["clip_l"], "sd1-vae", 512, 512, 7, 20));
        Register(models, new("stable-diffusion-v2-512", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, ["clip_g"], "sd1-vae", 512, 512, 7, 20));
        Register(models, new("stable-diffusion-v2-768-v", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.VPrediction, ["clip_g"], "sd1-vae", 768, 768, 7, 20));
        Register(models, new("stable-diffusion-xl-v1", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, ["clip_l", "clip_g"], "sdxl-vae", 1024, 1024, 7, 20));
        Register(models, new("segmind-stable-diffusion-1b", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, ["clip_l", "clip_g"], "sdxl-vae", 1024, 1024, 7, 20));
        Register(models, new("stable-video-diffusion-img2vid-v1", ModelType.ImageToVideo, ModelArchitecture.UNet, PredictionType.Epsilon, [], null, 1024, 576, 2.5, 25, DefaultScheduler: "karras", DefaultFrames: 14));

        // ================= DiT Models =================
        Register(models, new("stable-diffusion-v3-medium", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip_l", "clip_g", "t5xxl"], "sd35-vae", 1024, 1024, 4, 20, "EmptySD3LatentImage", DefaultScheduler: "sgm_uniform", SigmaShiftNode: "ModelSamplingSD3", DefaultSigmaShift: 3.0, ClipType: "sd3"));
        Register(models, new("stable-diffusion-v3.5-medium", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip_l", "clip_g", "t5xxl"], "sd35-vae", 1024, 1024, 4, 20, "EmptySD3LatentImage", DefaultScheduler: "sgm_uniform", SigmaShiftNode: "ModelSamplingSD3", DefaultSigmaShift: 3.0, ClipType: "sd3"));
        Register(models, new("stable-diffusion-v3-large", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip_l", "clip_g", "t5xxl"], "sd35-vae", 1024, 1024, 4, 20, "EmptySD3LatentImage", DefaultScheduler: "sgm_uniform", SigmaShiftNode: "ModelSamplingSD3",  DefaultSigmaShift: 3.0, ClipType: "sd3"));
        Register(models, new("stable-diffusion-v3.5-large", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip_l", "clip_g", "t5xxl"], "sd35-vae", 1024, 1024, 4, 20, "EmptySD3LatentImage", DefaultScheduler: "sgm_uniform", SigmaShiftNode: "ModelSamplingSD3", DefaultSigmaShift: 3.0, ClipType: "sd3"));
        Register(models, new("flux-1-dev", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip_l", "t5xxl"], "flux-ae", 1024, 1024, 1, 20, "EmptySD3LatentImage", ClipType: "flux"));
        Register(models, new("flux-1-schnell", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip_l", "t5xxl"], "flux-ae", 1024, 1024, 1, 4, "EmptySD3LatentImage", ClipType: "flux"));
        Register(models, new("flux-1-kontext", ModelType.ImageEdit, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip_l", "t5xxl"], "flux-ae", 1024, 1024, 1, 20, "EmptySD3LatentImage", ClipType: "flux"));
        Register(models, new("pixart-ms-sigma-xl-2", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, ["t5xxl"], "sdxl-vae", 1024, 1024, 4, 14, UniqueModelLoader: "PixArtCheckpointLoader", ClipType: "sd3"));
        Register(models, new("pixart-ms-sigma-xl-2-2k", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, ["t5xxl"], "sdxl-vae", 2048, 2048, 4, 14, UniqueModelLoader: "PixArtCheckpointLoader", ClipType: "sd3"));
        Register(models, new("nvidia-sana-1600", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, [], "sana-dcae", 1024, 1024, 7, 20, LatentNode: "EmptySanaLatentImage", UniqueClipLoader: "GemmaLoader", UniqueModelLoader: "SanaCheckpointLoader"));
        Register(models, new("auraflow-v1", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip_l", "clip_g", "t5xxl"], "flux-ae", 1024, 1024, 3.5, 20, DefaultScheduler: "sgm_uniform", DefaultSigmaShift: 1.73, SigmaShiftNode: "ModelSamplingAuraFlow", ClipType: "sd3"));
        Register(models, new("lumina-2", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["gemma2-2b"], "flux-ae", 1024, 1024, 4, 25, DefaultSampler: "res_multistep", DefaultSigmaShift: 6.0, ClipType: "lumina2"));
        Register(models, new("hidream-i1", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l-hidream", "clip-g-hidream", "t5xxl", "llama3.1_8b"], "flux-ae", 1024, 1024, 5, 50, LatentNode: "EmptySD3LatentImage", DefaultSampler: "uni_pc", SigmaShiftNode: "ModelSamplingSD3", DefaultSigmaShift: 3.0));
        Register(models, new("hidream-i1-edit", ModelType.ImageEdit, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l-hidream", "clip-g-hidream", "t5xxl", "llama3.1_8b"], "flux-ae", 768, 768, 5, 28, LatentNode: "EmptySD3LatentImage", DefaultScheduler: "normal", SigmaShiftNode: "ModelSamplingSD3", DefaultSigmaShift: 3.0));
        Register(models, new("nvidia-cosmos-predict2", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, ["old_t5xxl"], "wan21-vae", 1024, 1024, 4, 30, ClipType: "cosmos"));
        Register(models, new("qwen-image", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["qwen-2.5-vl-7b"], "qwen-image-vae", 1328, 1328, 2.5, 20, "EmptySD3LatentImage", DefaultSigmaShift: 3.1, SigmaShiftNode: "ModelSamplingAuraFlow", ClipType: "qwen_image"));
        Register(models, new("qwen-image-edit", ModelType.ImageEdit, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["qwen-2.5-vl-7b"], "qwen-image-vae", 1328, 1328, 2.5, 20, "EmptySD3LatentImage", DefaultSigmaShift: 3.0, SigmaShiftNode: "ModelSamplingAuraFlow", ClipType: "qwen_image"));
        Register(models, new("hunyuan-image-2_1", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["qwen-2.5-vl-7b", "byt5-small-glyphxl"], "hunyuan_image_2_1_vae", 2048, 2048, 3.5, 20, "EmptyHunyuanImageLatent", SigmaShiftNode: "ModelSamplingSD3", DefaultSigmaShift: 3.0, ClipType: "hunyuan_image"));
        Register(models, new("hunyuan-video", ModelType.TextToVideo, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip_l", "llava-llama3"], "hunyuan-video-vae", 720, 720, 6, 20, "EmptyHunyuanLatentVideo", DefaultFrames: 73, SigmaShiftNode: "ModelSamplingSD3",  DefaultSigmaShift: 7.0, ClipType: "hunyuan_video"));
        Register(models, new("wan-21", ModelType.TextAndImageToVideo, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["umt5xxl"], "wan21-vae", 960, 960, 6, 20, LatentNode: "EmptyHunyuanLatentVideo", DefaultScheduler: "simple", DefaultFrames: 81, DefaultSigmaShift: 3.0, SigmaShiftNode: "ModelSamplingSD3", ClipType: "wan"));
        Register(models, new("wan-22", ModelType.TextAndImageToVideo, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["umt5xxl"], "wan22-vae", 960, 960, 5, 20, LatentNode: "Wan22ImageToVideoLatent", DefaultScheduler: "simple", DefaultFrames: 81, DefaultSigmaShift: 3.0, SigmaShiftNode: "ModelSamplingSD3", ClipType: "wan"));
        Register(models, new("nvidia-cosmos-1", ModelType.TextAndImageToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, ["old_t5xxl"], "cosmos-vae", 960, 960, 6.5, 20, LatentNode: "EmptyCosmosLatentVideo", DefaultSampler: "res_multistep", DefaultScheduler: "karras", DefaultFrames: 121, ClipType: "cosmos"));
        Register(models, new("chroma", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["t5xxl"], "flux-ae", 1024, 1024, 3.5, 26, LatentNode: "EmptySD3LatentImage", DefaultScheduler: "beta", SigmaShiftNode: "ModelSamplingSD3", DefaultSigmaShift: 3.0, ClipType: "chroma"));
        Register(models, new("lightricks-ltx-video", ModelType.TextAndImageToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, ["t5xxl"], "ltxv-vae", 768, 512, 3, 30, LatentNode: "EmptyLTXVLatentVideo", DefaultScheduler: "ltxv", DefaultFrames: 97, ClipType: "ltxv"));
        Register(models, new("genmo-mochi-1", ModelType.TextToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, ["t5xxl"], "mochi-vae", 848, 480, 7, 20, LatentNode: "EmptyMochiLatentVideo", DefaultFrames: 25, ClipType: "mochi"));

        // ================= Cascade Models =================
        Register(models, new("stable-cascade-v1", ModelType.TextToImage, ModelArchitecture.Cascade, PredictionType.RectifiedFlow, [], null, 1024, 1024, 4, 20, LatentNode: "StableCascade_EmptyLatentImage", DefaultSampler: "euler_ancestral", DefaultScheduler: "simple"));

        // ================= MLLM Models =================
        Register(models, new("omnigen-2", ModelType.ImageEdit, ModelArchitecture.MLLM, PredictionType.RectifiedFlow, ["qwen-2.5-vl-fp16"], "flux-ae", 1024, 1024, 7, 20, LatentNode: "EmptySD3LatentImage"));

        Models = models.ToFrozenDictionary();
    }

    /// <summary>Gets the ModelInfo for a given compatibility class, or null if not found.</summary>
    public static ModelInfo GetModel(string baseModel) =>
        !string.IsNullOrEmpty(baseModel) && Models is not null && Models.TryGetValue(baseModel, out var modelInfo) ? modelInfo : null;
}