using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace SwarmUI.Text2Image;

/// <summary>Represents a class of models (eg SDv1).</summary>
public record class T2IModelClass
{
    /// <summary>Standard resolution for this model class.</summary>
    public int StandardWidth, StandardHeight;

    /// <summary>ID of this model type.</summary>
    public string ID;

    /// <summary>A clean name for this model class.</summary>
    public string Name;

    /// <summary>An identifier for a compatibility-class this class falls within (eg all SDv1 classes have the same compat class).</summary>
    public T2IModelCompatClass CompatClass;

    /// <summary>Matcher, return true if the model x safetensors header is the given class, or false if not.</summary>
    public Func<T2IModel, JObject, bool> IsThisModelOfClass;

    /// <summary>Get a networkable JObject for this model class.</summary>
    public JObject ToNetData()
    {
        return new JObject()
        {
            ["id"] = ID,
            ["name"] = Name,
            ["compat_class"] = CompatClass?.ID,
            ["standard_width"] = StandardWidth,
            ["standard_height"] = StandardHeight,
        };
    }
}

public enum ModelType
{
    TextToImage,
    TextToVideo,
    ImageToVideo,
    TextAndImageToVideo,
    ImageEdit,
    Refiner
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
    eps,
    v_prediction,
    sd3,
}

public record class T2IModelCompatClass
{
    /// <summary>ID of this model compat type.</summary>
    public string ID;

    /// <summary>A short label for this compat class, usually 4 letters long (but not always), used for quick previewing model types in UI.</summary>
    public string ShortCode;

    /// <summary>What kind of model is this (T2I, T2V, etc).</summary>
    public ModelType ModelType;

    /// <summary>The fundamental architecture of the model (UNet, DiT, etc).</summary>
    public ModelArchitecture Architecture;

    /// <summary>The prediction type for the model (eps, v_prediction, sd3).</summary>
    public PredictionType PredType;

    /// <summary>A list of required text encoder model IDs.</summary>
    public List<string> TextEncoders = [];

    /// <summary>The CLIP variant used by the model.</summary>
    public string ClipType;

    /// <summary>The default VAE model ID.</summary>
    public string VAE;

    /// <summary>The default latent image node used by this model architecture.</summary>
    public string LatentNode = "EmptyLatentImage";

    /// <summary>The default sigma shift node used by this model architecture (for SD3-like models).</summary>
    public string SigmaShiftNode = "ModelSamplingSD3";

    /// <summary>Default generation parameters for this model type.</summary>
    public List<string> DefaultParameters = ["steps:20", "cfgscale:7", "sampler:euler", "scheduler:normal"];

    /// <summary>If true, loras may target the text encoder. If false, they never do.</summary>
    public bool LorasTargetTextEnc = true;

    /// <summary>Get a networkable JObject for this compat class.</summary>
    public JObject ToNetData()
    {
        return new JObject()
        {
            ["id"] = ID,
            ["short_code"] = ShortCode,
            ["model_type"] = ModelType.ToString(),
            ["architecture"] = Architecture.ToString(),
            ["prediction_type"] = PredType.ToString(),
            ["text_encoders"] = new JArray(TextEncoders),
            ["clip_type"] = ClipType,
            ["vae"] = VAE,
            ["default_parameters"] = new JArray(DefaultParameters),
            ["loras_target_text_enc"] = LorasTargetTextEnc
        };
    }
}
