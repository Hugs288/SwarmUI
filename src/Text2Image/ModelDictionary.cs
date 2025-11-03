using System.Collections.Frozen;
using Microsoft.AspNetCore.Components.Web;

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
    string ClipType,
    string VAE,
    string LatentNode = "EmptyLatentImage",
    string SigmaShiftNode = null,
    List<string> DefaultParameters = null
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
        Register(models, new("stable-diffusion-v1", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, ["clip-l"], "stable-diffusion", "sd1-vae", DefaultParameters: ["sidelength:512", "aspectratio:1:1", "cfgscale:7", "steps:20"]));
        Register(models, new("stable-diffusion-v2-512", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, ["clip-g"], "stable-diffusion", "sd1-vae", DefaultParameters: ["sidelength:512", "aspectratio:1:1", "cfgscale:7", "steps:20"]));
        Register(models, new("stable-diffusion-v2-768-v", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.VPrediction, ["clip-g"], "stable-diffusion", "sd1-vae", DefaultParameters: ["sidelength:768", "aspectratio:1:1", "cfgscale:7", "steps:20"]));
        Register(models, new("stable-diffusion-xl-v1", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, ["clip-l", "clip-g"], "stable-diffusion", "sdxl-vae", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:7", "steps:20"]));
        Register(models, new("segmind-stable-diffusion-1b", ModelType.TextToImage, ModelArchitecture.UNet, PredictionType.Epsilon, ["clip-l", "clip-g"], "stable-diffusion", "sdxl-vae", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:7", "steps:20"]));
        Register(models, new("stable-video-diffusion-img2vid-v1", ModelType.ImageToVideo, ModelArchitecture.UNet, PredictionType.Epsilon, [], null, "stable-diffusion", DefaultParameters: ["sidelength:768", "aspectratio:16:9", "cfgscale:2.5", "steps:25", "text2videoframes:14", "scheduler:karras"]));

        // ================= DiT Models =================
        Register(models, new("stable-diffusion-v3-medium", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l", "clip-g", "t5xxl"], "sd3", "sd35-vae", "EmptySD3LatentImage", "ModelSamplingSD3", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:4", "steps:20", "scheduler:sgm_uniform", "sigmashift:3.0",]));
        Register(models, new("stable-diffusion-v3.5-medium", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l", "clip-g", "t5xxl"], "sd3", "sd35-vae", "EmptySD3LatentImage", "ModelSamplingSD3", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:4", "steps:20", "scheduler:sgm_uniform", "sigmashift:3.0",]));
        Register(models, new("stable-diffusion-v3-large", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l", "clip-g", "t5xxl"], "sd3", "sd35-vae", "EmptySD3LatentImage", "ModelSamplingSD3", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:4", "steps:20", "scheduler:sgm_uniform", "sigmashift:3.0",]));
        Register(models, new("stable-diffusion-v3.5-large", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l", "clip-g", "t5xxl"], "sd3", "sd35-vae", "EmptySD3LatentImage", "ModelSamplingSD3", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:4", "steps:20", "scheduler:sgm_uniform", "sigmashift:3.0",]));
        Register(models, new("flux-1-dev", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l", "t5xxl"], "flux", "flux-ae", "EmptySD3LatentImage", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:1", "steps:20"]));
        Register(models, new("flux-1-schnell", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l", "t5xxl"], "flux", "flux-ae", "EmptySD3LatentImage", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:1", "steps:4"]));
        Register(models, new("flux-1-kontext", ModelType.ImageEdit, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l", "t5xxl"], "flux", "flux-ae", "EmptySD3LatentImage", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:1", "steps:20, resizeimageprompts:1024"]));
        Register(models, new("pixart-ms-sigma-xl-2", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, ["t5xxl"], "sd3", "sdxl-vae", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:4", "steps:14"]));
        Register(models, new("pixart-ms-sigma-xl-2-2k", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, ["t5xxl"], "sd3", "sdxl-vae", DefaultParameters: ["sidelength:2048", "aspectratio:1:1", "cfgscale:4", "steps:14"]));
        Register(models, new("nvidia-sana-1600", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, [], null, "sana-dcae", "EmptySanaLatentImage", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:7", "steps:20"]));
        Register(models, new("auraflow-v1", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["pile-t5xxl"], "chroma", "sdxl-vae", "ModelSamplingAuraFlow", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:3.5", "steps:20", "sigmashift:1.73"]));
        Register(models, new("lumina-2", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["gemma2-2b"], "lumina", "flux-ae", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:4", "steps:25", "sampler:res_multistep", "sigmashift:6.0"]));
        Register(models, new("hidream-i1", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l-hidream", "clip-g-hidream", "t5xxl", "llama3.1-8b"], "hidream", "flux-ae", "EmptySD3LatentImage", "ModelSamplingSD3", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:5", "steps:50", "sampler:uni_pc", "sigmashift:3.0"]));
        Register(models, new("hidream-i1-edit", ModelType.ImageEdit, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l-hidream", "clip-g-hidream", "t5xxl", "llama3.1-8b"], "hidream", "flux-ae", "EmptySD3LatentImage", "ModelSamplingSD3", DefaultParameters: ["sidelength:768", "aspectratio:1:1", "cfgscale:5", "steps:28", "scheduler:normal", "sigmashift:3.0",]));
        Register(models, new("nvidia-cosmos-predict2", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.Epsilon, ["old-t5xxl"], "cosmos", "wan21-vae", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:4", "steps:30"]));
        Register(models, new("qwen-image", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["qwen-2.5-vl-7b"],  "qwen_image", "qwen-image-vae", "EmptySD3LatentImage", "ModelSamplingAuraFlow", DefaultParameters: ["sidelength:1328", "aspectratio:1:1", "cfgscale:2.5", "steps:20", "sigmashift:3.1"]));
        Register(models, new("qwen-image-edit", ModelType.ImageEdit, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["qwen-2.5-vl-7b"],  "qwen_image", "qwen-image-vae", "EmptySD3LatentImage", "ModelSamplingAuraFlow", DefaultParameters: ["sidelength:1328", "aspectratio:1:1", "cfgscale:2.5", "steps:20", "sigmashift:3.0", "resizeimageprompts:1024"]));
        Register(models, new("hunyuan-image-2_1", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["qwen-2.5-vl-7b", "byt5-small-glyphxl"], "hunyuan_image", "hunyuan_image_2_1_vae", "EmptyHunyuanImageLatent", "ModelSamplingSD3", DefaultParameters: ["sidelength:2048", "aspectratio:1:1", "cfgscale:3.5", "steps:20", "sigmashift:3.0"]));
        Register(models, new("hunyuan-video", ModelType.TextToVideo, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["clip-l", "llava-llama3"], "hunyuan_video", "hunyuan-video-vae", "EmptyHunyuanLatentVideo", "ModelSamplingSD3", DefaultParameters: ["sidelength:720", "aspectratio:1:1", "cfgscale:6", "steps:20", "text2videoframes:73", "sigmashift:7.0"]));
        Register(models, new("wan-21-1_3b_t2v", ModelType.TextToVideo, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["umt5xxl"], "wan", "wan21-vae", "EmptyHunyuanLatentVideo", "ModelSamplingSD3", DefaultParameters: ["sidelength:480", "aspectratio:1:1", "cfgscale:6", "steps:30", "text2videoframes:81", "sampler:uni_pc", "sigmashift:8"]));
        Register(models, new("wan-21-1_3b_i2v", ModelType.ImageToVideo, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["umt5xxl"], "wan", "wan21-vae", "EmptyHunyuanLatentVideo", "ModelSamplingSD3", DefaultParameters: ["sidelength:480", "aspectratio:1:1", "cfgscale:6", "steps:30", "text2videoframes:81", "sampler:uni_pc", "sigmashift:8"]));
        Register(models, new("wan-21-14b-t2v", ModelType.TextToVideo, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["umt5xxl"], "wan", "wan21-vae", "EmptyHunyuanLatentVideo", "ModelSamplingSD3", DefaultParameters: ["sidelength:720", "aspectratio:1:1", "cfgscale:6", "steps:30", "text2videoframes:81", "sampler:uni_pc", "sigmashift:8"]));
        Register(models, new("wan-21-14b-i2v", ModelType.ImageToVideo, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["umt5xxl"], "wan", "wan21-vae", "EmptyHunyuanLatentVideo", "ModelSamplingSD3", DefaultParameters: ["sidelength:720", "aspectratio:1:1", "cfgscale:6", "steps:30", "text2videoframes:81", "sampler:uni_pc", "sigmashift:8"]));
        Register(models, new("wan-22", ModelType.TextAndImageToVideo, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["umt5xxl"], "wan", "wan22-vae", "Wan22ImageToVideoLatent", "ModelSamplingSD3", DefaultParameters: ["sidelength:960", "aspectratio:1:1", "cfgscale:5", "steps:20", "text2videoframes:81", "sigmashift:3.0"]));
        Register(models, new("nvidia-cosmos-1", ModelType.TextAndImageToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, ["old-t5xxl"], "cosmos", "cosmos-vae", "EmptyCosmosLatentVideo", DefaultParameters: ["sidelength:960", "aspectratio:1:1", "cfgscale:6.5", "steps:20", "sampler:res_multistep", "scheduler:karras", "text2videoframes:121"]));
        Register(models, new("chroma", ModelType.TextToImage, ModelArchitecture.Dit, PredictionType.RectifiedFlow, ["t5xxl"], "chroma", "flux-ae", "EmptySD3LatentImage", "ModelSamplingSD3", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:3.5", "steps:26", "scheduler:beta", "sigmashift:3.0",]));
        Register(models, new("lightricks-ltx-video", ModelType.TextAndImageToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, ["t5xxl"], "ltxv", "ltxv-vae", "EmptyLTXVLatentVideo", DefaultParameters: ["sidelength:768", "aspectratio:2:3", "cfgscale:3", "steps:30", "scheduler:ltxv", "text2videoframes:97",]));
        Register(models, new("genmo-mochi-1", ModelType.TextToVideo, ModelArchitecture.Dit, PredictionType.Epsilon, ["t5xxl"], "mochi", "mochi-vae", "EmptyMochiLatentVideo", DefaultParameters: ["sidelength:640", "aspectratio:16:9", "cfgscale:7", "steps:20", "text2videoframes:25"]));

        // ================= Cascade Models =================
        Register(models, new("stable-cascade-v1", ModelType.TextToImage, ModelArchitecture.Cascade, PredictionType.RectifiedFlow, [], "stable-cascade", null, "StableCascade_EmptyLatentImage", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:4", "steps:20", "sampler:euler_ancestral", "cascadelatentcompression:32"]));

        // ================= MLLM Models =================
        Register(models, new("omnigen-2", ModelType.ImageEdit, ModelArchitecture.MLLM, PredictionType.RectifiedFlow, ["qwen-2.5-vl-fp16"], "omnigen2", "flux-ae", "EmptySD3LatentImage", DefaultParameters: ["sidelength:1024", "aspectratio:1:1", "cfgscale:7", "steps:20"]));

        Models = models.ToFrozenDictionary();
    }

    /// <summary>Gets the ModelInfo for a given compatibility class, or null if not found.</summary>
    public static ModelInfo GetModel(string baseModel) =>
        !string.IsNullOrEmpty(baseModel) && Models is not null && Models.TryGetValue(baseModel, out var modelInfo) ? modelInfo : null;
}