using System.IO;
using System.Runtime.InteropServices;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using Newtonsoft.Json.Linq;
using FreneticUtilities.FreneticExtensions;

namespace SwarmUI.Builtin_ComfyUIBackend;

public partial class WorkflowGenerator
{
    /// <summary>
    /// Map of model architecture IDs to Func(int width, int height, int batchSize, string id = null) => string NodeID.
    /// Used for custom model classes to implement <see cref="CreateEmptyImage"/>
    /// </summary>
    public static Dictionary<string, Func<int, int, int, string, string>> EmptyImageCreators = [];

    public bool IsVideoModel()
    {
        return CurrentCompatClass() is "lightricks-ltx-video" or "genmo-mochi-1" or "hunyuan-video" or "hunyuan-video-1_5" or "nvidia-cosmos-1" || CurrentCompatClass().StartsWith("wan-21") || CurrentCompatClass().StartsWith("wan-22");
    }

    /// <summary>Creates an Empty Latent Image node.</summary>
    public string CreateEmptyImage(int width, int height, int batchSize, string id = null)
    {
        if (EmptyImageCreators.TryGetValue(CurrentModelClass()?.ID, out Func<int, int, int, string, string> creator))
        {
            return creator(width, height, batchSize, id);
        }
        if (CurrentModelInfo().Architecture.ToString() == "UNet" && UserInput.Get(ComfyUIBackendExtension.ShiftedLatentAverageInit, false))
        {
            double offA = 0, offB = 0, offC = 0, offD = 0;
            switch (FinalLoadedModel.ModelClass?.CompatClass?.ID)
            {
                case "stable-diffusion-v1": // https://github.com/Birch-san/sdxl-diffusion-decoder/blob/4ba89847c02db070b766969c0eca3686a1e7512e/script/inference_decoder.py#L112
                case "stable-diffusion-v2-512":
                case "stable-diffusion-v2-768-v":
                    offA = 2.1335;
                    offB = 0.1237;
                    offC = 0.4052;
                    offD = -0.0940;
                    break;
                case "stable-diffusion-xl-v1": // https://huggingface.co/datasets/Birchlabs/sdxl-latents-ffhq
                    offA = -2.8982;
                    offB = -0.9609;
                    offC = 0.2416;
                    offD = -0.3074;
                    break;
            }
            return CreateNode("SwarmOffsetEmptyLatentImage", new JObject()
            {
                ["batch_size"] = batchSize,
                ["height"] = height,
                ["width"] = width,
                ["off_a"] = offA,
                ["off_b"] = offB,
                ["off_c"] = offC,
                ["off_d"] = offD
            }, id);
        }
        string latentNode = CurrentModelInfo().LatentNode;
        JObject inputs = new()
        {
            ["batch_size"] = batchSize,
            ["height"] = height,
            ["width"] = width
        };
        if (CurrentModelInfo().ModelType is ModelType.TextToVideo or ModelType.ImageToVideo or ModelType.TextAndImageToVideo)
        {
            inputs["length"] = UserInput.Get(T2IParamTypes.Text2VideoFrames, 25);
        }
        if (CurrentModelInfo().Architecture == ModelArchitecture.Cascade)
        {
            inputs["compression"] = UserInput.Get(T2IParamTypes.CascadeLatentCompression, 32);
        }
        if (latentNode == "Wan22ImageToVideoLatent")
        {
            inputs["vae"] = FinalVae;
        }
        return CreateNode(latentNode, inputs, id);
    }

    public class ModelLoadHelpers(WorkflowGenerator g)
    {
        public void DoVaeLoader(string compatClass, string knownName)
        {
            string vaeNameToLoad = null;
            string nodeId = null;
            // 1. Check for a user-specified VAE first.
            if (g.UserInput.TryGet(T2IParamTypes.VAE, out T2IModel vaeModel))
            {
                vaeNameToLoad = vaeModel.Name;
                nodeId = "11";
            }
            else
            {
                // 2. If no user VAE is provided, attempt to find a suitable one automatically.
                CommonModels.ModelInfo knownFile = knownName is null ? null : CommonModels.Known.GetValueOrDefault(knownName);
                // 2a. Check if the recommended 'knownFile' is already installed.
                if (knownFile is not null && Program.T2IModelSets["VAE"].Models.ContainsKey(knownFile.FileName))
                {
                    vaeNameToLoad = knownFile.FileName;
                }
                else
                {
                    // 2b. If not, find any other installed VAE with a matching compatibility class.
                    T2IModel matchingVae = Program.T2IModelSets["VAE"].Models.Values.FirstOrDefault(m => m.ModelClass?.CompatClass?.ID == compatClass);
                    if (matchingVae is not null)
                    {
                        Logs.Debug($"Auto-selected first available VAE of compat class '{compatClass}', VAE '{matchingVae.Name}' will be applied");
                        vaeNameToLoad = matchingVae.Name;
                    }
                    // 2c. If no compatible VAE is found locally and a downloadable 'knownFile' exists, download it.
                    else if (knownFile is not null)
                    {
                        vaeNameToLoad = knownFile.FileName;
                        knownFile.DownloadNow().Wait();
                        Program.RefreshAllModelSets();
                    }
                }
            }
            // 3. If no VAE could be determined, throw an error.
            if (string.IsNullOrWhiteSpace(vaeNameToLoad))
            {
                throw new SwarmUserErrorException("No default VAE for this model found, this normally should not happen, but you can select one to use.");
            }
            // 4. Create the VAE loader node in the workflow.
            g.LoadingVAE = g.CreateVAELoader(vaeNameToLoad, nodeId);
        }

        public string DoClipLoader(string id, T2IRegisteredParam<T2IModel> param)
        {
            if (param is not null && g.UserInput.TryGet(param, out T2IModel model))
            {
                return model.Name;
            }
            if (!CommonModels.Known.TryGetValue(id, out CommonModels.ModelInfo info))
            {
                throw new InvalidOperationException($"Unknown common model ID: {id}");
            }
            string name = info.FileName;
            if (ClipModelsValid.ContainsKey(name))
            {
                return name;
            }
            if (Program.T2IModelSets[info.FolderType].Models.ContainsKey(name))
            {
                ClipModelsValid.TryAdd(name, name);
                return name;
            }
            g.EnsureCommonModel(info);
            Program.RefreshAllModelSets();
            ClipModelsValid.TryAdd(name, name);
            return name;
        }
    }

    /// <summary>Creates a model loader and adapts it with any registered model adapters, and returns (Model, Clip, VAE).</summary>
    public (T2IModel, JArray, JArray, JArray) CreateStandardModelLoader(T2IModel model, string type, string id = null, bool noCascadeFix = false)
    {
        ModelLoadHelpers helpers = new(this);
        string helper = $"modelloader_{model.Name}_{type}";
        if (NodeHelpers.TryGetValue(helper, out string alreadyLoaded))
        {
            string[] parts = alreadyLoaded.SplitFast(':');
            LoadingModel = [parts[0], int.Parse(parts[1])];
            LoadingClip = parts[2].Length == 0 ? null : [parts[2], int.Parse(parts[3])];
            LoadingVAE = parts[4].Length == 0 ? null : [parts[4], int.Parse(parts[5])];
            return (model, LoadingModel, LoadingClip, LoadingVAE);
        }
        IsDifferentialDiffusion = false;
        LoadingModelType = type;
        if (!noCascadeFix && model.ModelClass?.ID == "stable-cascade-v1-stage-b" && model.Name.Contains("stage_b") && Program.MainSDModels.Models.TryGetValue(model.Name.Replace("stage_b", "stage_c"), out T2IModel altCascadeModel))
        {
            model = altCascadeModel;
        }
        LoadingModel = null;
        foreach (WorkflowGenStep step in ModelGenSteps.Where(s => s.Priority <= -100))
        {
            step.Action(this);
        }
        if (LoadingModel is not null)
        {
            // Custom action has loaded it for us.
        }
        else if (model.ModelClass?.ID.EndsWith("/tensorrt") ?? false)
        {
            string baseArch = model.ModelClass?.ID?.Before('/');
            string trtType = ComfyUIWebAPI.ArchitecturesTRTCompat[baseArch];
            string trtloader = CreateNode("TensorRTLoader", new JObject()
            {
                ["unet_name"] = model.ToString(ModelFolderFormat),
                ["model_type"] = trtType
            }, id);
            LoadingModel = [trtloader, 0];
            // TODO: This is a hack
            T2IModel[] sameArch = [.. Program.MainSDModels.Models.Values.Where(m => m.ModelClass?.ID == baseArch)];
            if (sameArch.Length == 0)
            {
                throw new SwarmUserErrorException($"No models found with architecture {baseArch}, cannot load CLIP/VAE for this Arch");
            }
            T2IModel matchedName = sameArch.FirstOrDefault(m => m.Name.Before('.') == model.Name.Before('.'));
            matchedName ??= sameArch.First();
            string secondaryNode = CreateNode("CheckpointLoaderSimple", new JObject()
            {
                ["ckpt_name"] = matchedName.ToString(ModelFolderFormat)
            });
            LoadingClip = [secondaryNode, 1];
            LoadingVAE = [secondaryNode, 2];
        }
        else if (model.Name.EndsWith(".engine"))
        {
            throw new SwarmUserErrorException($"Model {model.Name} appears to be TensorRT lacks metadata to identify its architecture, cannot load");
        }
        else if (model.ModelClass?.CompatClass?.ID == "pixart-ms-sigma-xl-2")
        {
            string pixartNode = CreateNode("PixArtCheckpointLoader", new JObject()
            {
                ["ckpt_name"] = model.ToString(ModelFolderFormat),
                ["model"] = model.ModelClass.ID == "pixart-ms-sigma-xl-2-2k" ? "PixArtMS_Sigma_XL_2_2K" : "PixArtMS_Sigma_XL_2"
            }, id);
            LoadingModel = [pixartNode, 0];
            string singleClipLoader = CreateNode("CLIPLoader", new JObject()
            {
                ["clip_name"] = helpers.DoClipLoader("t5xxl", T2IParamTypes.T5XXLModel),
                ["type"] = "sd3"
            });
            LoadingClip = [singleClipLoader, 0];
            helpers.DoVaeLoader("stable-diffusion-xl-v1", "sdxl-vae");
        }
        else if (model.IsDiffusionModelsFormat)
        {
            if (model.Metadata?.SpecialFormat is "gguf")
            {
                if (!Features.Contains("gguf"))
                {
                    throw new SwarmUserErrorException($"Model '{model.Name}' is in GGUF format, but the server does not have GGUF support installed. Cannot run.");
                }
                string modelNode = CreateNode("UnetLoaderGGUF", new JObject()
                {
                    ["unet_name"] = model.ToString(ModelFolderFormat)
                }, id);
                LoadingModel = [modelNode, 0];
            }
            else if (model.Metadata?.SpecialFormat is "nunchaku" or "nunchaku-fp4")
            {
                if (!Features.Contains("nunchaku"))
                {
                    throw new SwarmUserErrorException($"Model '{model.Name}' is in Nunchaku format, but the server does not have Nunchaku support installed. Cannot run.");
                }
                if (CurrentCompatClass().StartsWith("flux-1"))
                {
                    // TODO: Configuration of these params?
                    string modelNode = CreateNode("NunchakuFluxDiTLoader", new JObject()
                    {
                        ["model_path"] = model.Name.EndsWith("/transformer_blocks.safetensors") ? model.Name.BeforeLast('/').Replace("/", ModelFolderFormat ?? $"{Path.DirectorySeparatorChar}") : model.ToString(ModelFolderFormat),
                        ["cache_threshold"] = UserInput.Get(ComfyUIBackendExtension.NunchakuCacheThreshold, 0),
                        ["attention"] = "nunchaku-fp16",
                        ["cpu_offload"] = "auto",
                        ["device_id"] = 0,
                        ["data_type"] = model.Metadata?.SpecialFormat == "nunchaku-fp4" ? "bfloat16" : "float16",
                        ["i2f_mode"] = "enabled"
                    }, id);
                    LoadingModel = [modelNode, 0];
                }
                else if (CurrentCompatClass().StartsWith("qwen-image"))
                {
                    string modelNode = CreateNode("NunchakuQwenImageDiTLoader", new JObject()
                    {
                        ["model_name"] = model.Name.EndsWith("/transformer_blocks.safetensors") ? model.Name.BeforeLast('/').Replace("/", ModelFolderFormat ?? $"{Path.DirectorySeparatorChar}") : model.ToString(ModelFolderFormat),
                        ["cpu_offload"] = "auto",
                        ["num_blocks_on_gpu"] = 1, // TODO: If nunchaku doesn't fix automation here, add a param. Also enable cpu_offload if the param is given.
                        ["use_pin_memory"] = "enable"
                    }, id);
                    LoadingModel = [modelNode, 0];
                }
                else
                {
                    throw new SwarmUserErrorException($"Cannot load nunchaku for model architecture '{model.ModelClass?.ID}'. If other model architectures are supported in the Nunchaku source, please report this on the SwarmUI GitHub or Discord.");
                }
            }
            else if (model.Metadata?.SpecialFormat is "bnb_nf4" or "bnb_fp4")
            {
                if (!Features.Contains("bnb_nf4"))
                {
                    throw new SwarmUserErrorException($"Model '{model.Name}' is in BitsAndBytes-NF4 format, but the server does not have BNB_NF4 support installed. Cannot run.");
                }
                string modelNode = CreateNode("UNETLoaderNF4", new JObject()
                {
                    ["unet_name"] = model.ToString(ModelFolderFormat),
                    ["bnb_dtype"] = model.Metadata?.SpecialFormat == "bnb_fp4" ? "fp4" : "nf4"
                }, id);
                LoadingModel = [modelNode, 0];
            }
            else
            {
                if (model.RawFilePath.EndsWith(".gguf"))
                {
                    Logs.Error($"Model '{model.Name}' likely has corrupt/invalid metadata, and needs to be reset.");
                }
                string dtype = UserInput.Get(ComfyUIBackendExtension.PreferredDType, "automatic");
                if (dtype == "automatic")
                {
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX) || model.Metadata?.SpecialFormat == "fp8_scaled" || CurrentCompatClass() is "nvidia-cosmos-predict2" or "chroma" or "chroma-radiance" or "z-image" || CurrentCompatClass().StartsWith("omnigen-")) // TODO: Or AMD?
                    {
                        dtype = "default";
                    }
                    else
                    {
                        dtype = "fp8_e4m3fn";
                        if (Utilities.PresumeNVidia30xx && Program.ServerSettings.Performance.AllowGpuSpecificOptimizations && !CurrentCompatClass().StartsWith("qwen-image"))
                        {
                            dtype = "fp8_e4m3fn_fast";
                        }
                    }
                }
                string modelNode = CreateNode("UNETLoader", new JObject()
                {
                    ["unet_name"] = model.ToString(ModelFolderFormat),
                    ["weight_dtype"] = dtype
                }, id);
                LoadingModel = [modelNode, 0];
            }
            LoadingClip = null;
            LoadingVAE = null;
        }
        else if (model.Metadata?.SpecialFormat is "bnb_nf4" or "bnb_fp4")
        {
            if (!Features.Contains("bnb_nf4"))
            {
                throw new SwarmUserErrorException($"Model '{model.Name}' is in BitsAndBytes-NF4 format, but the server does not have BNB_NF4 support installed. Cannot run.");
            }
            string modelNode = CreateNode("CheckpointLoaderNF4", new JObject()
            {
                ["ckpt_name"] = model.ToString(ModelFolderFormat),
                ["bnb_dtype"] = model.Metadata?.SpecialFormat == "bnb_fp4" ? "fp4" : "nf4"
            }, id);
            LoadingModel = [modelNode, 0];
            LoadingClip = [modelNode, 1];
            LoadingVAE = [modelNode, 2];
        }
        else if (CurrentCompatClass() is "nvidia-sana-1600")
        {
            string sanaNode = CreateNode("SanaCheckpointLoader", new JObject()
            {
                ["ckpt_name"] = model.ToString(ModelFolderFormat),
                ["model"] = "SanaMS_1600M_P1_D20"
            }, id);
            LoadingModel = [sanaNode, 0];
            string clipLoader = CreateNode("GemmaLoader", new JObject()
            {
                ["model_name"] = "unsloth/gemma-2-2b-it-bnb-4bit",
                ["device"] = "cpu",
                ["dtype"] = "default"
            });
            LoadingClip = [clipLoader, 0];
            helpers.DoVaeLoader("nvidia-sana-1600", "sana-dcae");
        }
        else if (CurrentCompatClass() is "pixart-ms-sigma-xl-2")
        {
            string pixartNode = CreateNode("PixArtCheckpointLoader", new JObject()
            {
                ["ckpt_name"] = model.ToString(ModelFolderFormat),
                ["model"] = model.ModelClass.ID == "pixart-ms-sigma-xl-2-2k" ? "PixArtMS_Sigma_XL_2_2K" : "PixArtMS_Sigma_XL_2"
            }, id);
            LoadingModel = [pixartNode, 0];
            string singleClipLoader = CreateNode("CLIPLoader", new JObject()
            {
                ["clip_name"] = helpers.DoClipLoader("t5xxl", T2IParamTypes.T5XXLModel),
                ["type"] = "sd3"
            });
            LoadingClip = [singleClipLoader, 0];
            helpers.DoVaeLoader("stable-diffusion-xl-v1", "sdxl-vae");
        }
        else
        {
            if (model.Metadata?.SpecialFormat is "gguf" or "nunchaku" or "nunchaku-fp4")
            {
                throw new SwarmUserErrorException($"Model '{model.Name}' is in '{model.Metadata.SpecialFormat}' format, but it's in your main Stable-Diffusion models folder. You should move it into the 'diffusion_models' folder.");
            }
            string modelNode = CreateNode("CheckpointLoaderSimple", new JObject()
            {
                ["ckpt_name"] = model.ToString(ModelFolderFormat)
            }, id);
            LoadingModel = [modelNode, 0];
            LoadingClip = [modelNode, 1];
            LoadingVAE = [modelNode, 2];
        }
        if (CurrentModelInfo() is not null)
        {
            // text encoders
            if (LoadingClip is null) {
                Dictionary<string, T2IRegisteredParam<T2IModel>> encoderParams = new()
                {
                    ["clip-l"] = T2IParamTypes.ClipLModel, ["clip-g"] = T2IParamTypes.ClipGModel, ["t5xxl"] = T2IParamTypes.T5XXLModel,
                    ["llava-llama3"] = T2IParamTypes.LLaVAModel, ["llama3.1-8b"] = T2IParamTypes.LLaMAModel, ["qwen-2.5-vl-7b"] = T2IParamTypes.QwenModel,
                    ["clip-l-hidream"] = T2IParamTypes.ClipLModel, ["clip-g-hidream"] = T2IParamTypes.ClipGModel, ["old-t5xxl"] = T2IParamTypes.T5XXLModel,
                    ["umt5xxl"] = T2IParamTypes.T5XXLModel, ["gemma2-2b"] = null, ["pile-t5xxl"] = T2IParamTypes.T5XXLModel, ["byt5-small-glyphxl"] = T2IParamTypes.T5XXLModel,
                    ["qwen-2.5-vl-fp16"] = T2IParamTypes.QwenModel
                };
                List<string> encoders = CurrentModelInfo().TextEncoders ?? [];
                string[] encoderFiles = encoders.Select(e => helpers.DoClipLoader(e, encoderParams.GetValueOrDefault(e))).ToArray();
                bool anyGguf = encoderFiles.Any(f => f.EndsWith(".gguf"));
                string loaderType = (encoders.Count, anyGguf) switch
                {
                    (1, false) => "CLIPLoader", (1, true) => "CLIPLoaderGGUF",
                    (2, false) => "DualCLIPLoader", (2, true) => "DualCLIPLoaderGGUF",
                    (3, false) => "TripleCLIPLoader", (3, true) => "TripleCLIPLoaderGGUF",
                    (4, false) => "QuadrupleCLIPLoader", (4, true) => "QuadrupleCLIPLoaderGGUF",
                    _ => throw new SwarmUserErrorException($"Unsupported text encoder count: {encoders.Count}")
                };
                JObject clipInputs = [];
                for (int i = 0; i < encoders.Count; i++)
                {
                    clipInputs[$"clip_name{(i == 0 ? "" : (i + 1))}"] = encoderFiles[i];
                }
                if (CurrentModelInfo().ClipType is not null)
                {
                    clipInputs["type"] = CurrentModelInfo().ClipType;
                }
                string clipLoaderNode = CreateNode(loaderType, clipInputs);
                LoadingClip = [clipLoaderNode, 0];
            }
            if (CurrentCompatClass() is "auraflow-v1" or "chroma" or "chroma-radiance")
            {
                string t5Patch = CreateNode("T5TokenizerOptions", new JObject()
                {
                    ["clip"] = LoadingClip,
                    ["min_padding"] = CurrentCompatClass() == "auraflow-v1" ? 768 : 0,
                    ["min_length"] = CurrentCompatClass() == "auraflow-v1" ? 768 : 0
                });
                LoadingClip = [t5Patch, 0];
            }
            // vae
            if (LoadingVAE is null)
            {
                if (CurrentCompatClass() == "chroma-radiance")
                {
                    LoadingVAE = CreateVAELoader("pixel_space");
                }
                else helpers.DoVaeLoader(CurrentCompatClass(), CurrentModelInfo().VAE);
            }
        }
        string predType = model.Metadata?.PredictionType == null ? CurrentModelInfo().PredType.ToString() : model.Metadata?.PredictionType;
        if (!string.IsNullOrWhiteSpace(predType) && LoadingModel is not null)
        {
            if (predType == "sd3")
            {
                if (CurrentModelInfo().SigmaShiftNode == "ModelSamplingFlux")
                {
                    string samplingNode = CreateNode("ModelSamplingFlux", new JObject()
                    {
                        ["model"] = LoadingModel,
                        ["width"] = UserInput.GetImageResolution().Width,
                        ["height"] = UserInput.GetImageResolution().Height,
                        ["max_shift"] = UserInput.Get(T2IParamTypes.SigmaShift, 3),
                        ["base_shift"] = 0.5 // TODO: Does this need an input?
                    });
                    LoadingModel = [samplingNode, 0];
                }
                else
                {
                    string samplingNode = CreateNode(CurrentModelInfo().SigmaShiftNode, new JObject()
                    {
                        ["model"] = LoadingModel,
                        ["shift"] = UserInput.Get(T2IParamTypes.SigmaShift, 3)
                    });
                    LoadingModel = [samplingNode, 0];
                }
            }
            else if (model.Metadata?.PredictionType != null && model.Metadata?.PredictionType != CurrentModelInfo().PredType.ToString())
            {
                string discreteNode = CreateNode("ModelSamplingDiscrete", new JObject()
                {
                    ["model"] = LoadingModel,
                    ["sampling"] = predType switch { "v" => "v_prediction", "v-zsnr" => "v_prediction", "epsilon" => "eps", _ => predType },
                    ["zsnr"] = predType.Contains("zsnr")
                });
                LoadingModel = [discreteNode, 0];
            }
        }
        foreach (WorkflowGenStep step in ModelGenSteps.Where(s => s.Priority > -100))
        {
            step.Action(this);
        }
        if (LoadingClip is null)
        {
            if (string.IsNullOrWhiteSpace(model.Metadata?.ModelClassType))
            {
                throw new SwarmUserErrorException($"Model loader for {model.Name} didn't work - architecture ID is missing. Please click Edit Metadata on the model and apply a valid architecture ID.");
            }
            throw new SwarmUserErrorException($"Model loader for {model.Name} didn't work - are you sure it has an architecture ID set properly? (Currently set to: '{model.Metadata?.ModelClassType}')");
        }
        NodeHelpers[helper] = $"{LoadingModel[0]}:{LoadingModel[1]}" + (LoadingClip is null ? "::" : $":{LoadingClip[0]}:{LoadingClip[1]}") + (LoadingVAE is null ? "::" : $":{LoadingVAE[0]}:{LoadingVAE[1]}");
        return (model, LoadingModel, LoadingClip, LoadingVAE);
    }
}
