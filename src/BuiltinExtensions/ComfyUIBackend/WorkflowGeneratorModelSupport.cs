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
        return CurrentCompatClass() is "lightricks-ltx-video" or "genmo-mochi-1" or "hunyuan-video" or "nvidia-cosmos-1" || CurrentCompatClass().StartsWith("wan-21") || CurrentCompatClass().StartsWith("wan-22");
    }

    /// <summary>Creates an Empty Latent Image node.</summary>
    public string CreateEmptyImage(int width, int height, int batchSize, string id = null)
    {
        if (EmptyImageCreators.TryGetValue(CurrentModelClass()?.ID, out Func<int, int, int, string, string> creator))
        {
            return creator(width, height, batchSize, id);
        }
        ModelInfo info = ModelDictionary.GetModel(CurrentCompatClass());
        if (info.Architecture.ToString() == "UNet" && UserInput.Get(ComfyUIBackendExtension.ShiftedLatentAverageInit, false))
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
        string latentNode = info.LatentNode;
        JObject inputs = new()
        {
            ["batch_size"] = batchSize,
            ["height"] = height,
            ["width"] = width
        };
        if (info.Type is ModelType.TextToVideo or ModelType.ImageToVideo or ModelType.TextAndImageToVideo)
        {
            inputs["length"] = UserInput.Get(T2IParamTypes.Text2VideoFrames, 25);
        }
        if (info.Architecture == ModelArchitecture.Cascade)
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
        public void DoVaeLoader(string defaultVal, string compatClass, string knownName)
        {
            string vaeFile = defaultVal;
            string nodeId = null;
            CommonModels.ModelInfo knownFile = knownName is null ? null : CommonModels.Known[knownName];
            if (!g.NoVAEOverride && g.UserInput.TryGet(T2IParamTypes.VAE, out T2IModel vaeModel))
            {
                vaeFile = vaeModel.Name;
                nodeId = "11";
            }
            if (vaeFile == "None")
            {
                vaeFile = null;
            }
            if (string.IsNullOrWhiteSpace(vaeFile) && knownFile is not null && Program.T2IModelSets["VAE"].Models.ContainsKey(knownFile.FileName))
            {
                vaeFile = knownFile.FileName;
            }
            if (string.IsNullOrWhiteSpace(vaeFile))
            {
                vaeModel = Program.T2IModelSets["VAE"].Models.Values.FirstOrDefault(m => m.ModelClass?.CompatClass?.ID == compatClass);
                if (vaeModel is not null)
                {
                    Logs.Debug($"Auto-selected first available VAE of compat class '{compatClass}', VAE '{vaeModel.Name}' will be applied");
                    vaeFile = vaeModel.Name;
                }
            }
            if (string.IsNullOrWhiteSpace(vaeFile))
            {
                if (knownFile is null)
                {
                    throw new SwarmUserErrorException("No default VAE for this model found, please download its VAE and set it as default in User Settings");
                }
                vaeFile = knownFile.FileName;
                knownFile.DownloadNow().Wait();
                Program.RefreshAllModelSets();
            }
            g.LoadingVAE = g.CreateVAELoader(vaeFile, nodeId);
        }

        string RequireClipModel(string name, string url, string hash, T2IRegisteredParam<T2IModel> param)
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
            string filePath = Utilities.CombinePathWithAbsolute(Program.ServerSettings.Paths.ActualModelRoot, Program.ServerSettings.Paths.SDClipFolder.Split(';')[0], name);
            g.DownloadModel(name, filePath, url, hash);
            ClipModelsValid.TryAdd(name, name);
            return name;
        }

        public string GetT5XXLModel()
        {
            return RequireClipModel("t5xxl_enconly.safetensors", "https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/resolve/main/t5xxl_fp8_e4m3fn.safetensors", "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09", T2IParamTypes.T5XXLModel);
        }

        public string GetOldT5XXLModel()
        {
            return RequireClipModel("old_t5xxl_cosmos.safetensors", "https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/resolve/main/text_encoders/oldt5_xxl_fp8_e4m3fn_scaled.safetensors", "1d0dd711ec9866173d4b39e86db3f45e1614a4e3f84919556f854f773352ea81", T2IParamTypes.T5XXLModel);
        }

        public string GetUniMaxT5XXLModel()
        {
            return RequireClipModel("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68", T2IParamTypes.T5XXLModel);
        }

        public string GetByT5SmallGlyphxl_tenc()
        {
            return RequireClipModel("byt5_small_glyphxl_fp16.safetensors", "https://huggingface.co/Comfy-Org/HunyuanImage_2.1_ComfyUI/resolve/main/split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", "516910bb4c9b225370290e40585d1b0e6c8cd3583690f7eec2f7fb593990fb48", T2IParamTypes.T5XXLModel);
        }

        public string GetPileT5XLAuraFlow()
        {
            return RequireClipModel("pile_t5xl_auraflow.safetensors", "https://huggingface.co/fal/AuraFlow-v0.2/resolve/main/text_encoder/model.safetensors", "0a07449cf1141c0ec86e653c00465f6f0d79c6e58a2c60c8bcf4203d0e4ec4f6", T2IParamTypes.T5XXLModel);
        }
        public string GetOmniQwenModel()
        {
            return RequireClipModel("qwen_2.5_vl_fp16.safetensors", "https://huggingface.co/Comfy-Org/Omnigen2_ComfyUI_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_fp16.safetensors", "ba05dd266ad6a6aa90f7b2936e4e775d801fb233540585b43933647f8bc4fbc3", T2IParamTypes.QwenModel);
        }

        public string GetQwenImage25_7b_tenc()
        {
            return RequireClipModel("qwen_2.5_vl_7b_fp8_scaled.safetensors", "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4", T2IParamTypes.QwenModel);
        }

        public string GetClipLModel()
        {
            if (g.UserInput.TryGet(T2IParamTypes.ClipLModel, out T2IModel model))
            {
                return model.Name;
            }
            if (Program.T2IModelSets["Clip"].Models.ContainsKey("clip_l_sdxl_base.safetensors"))
            {
                return "clip_l_sdxl_base.safetensors";
            }
            return RequireClipModel("clip_l.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.fp16.safetensors", "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd", T2IParamTypes.ClipLModel);
        }

        public string GetClipGModel()
        {
            if (g.UserInput.TryGet(T2IParamTypes.ClipGModel, out T2IModel model))
            {
                return model.Name;
            }
            if (Program.T2IModelSets["Clip"].Models.ContainsKey("clip_g_sdxl_base.safetensors"))
            {
                return "clip_g_sdxl_base.safetensors";
            }
            return RequireClipModel("clip_g.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.fp16.safetensors", "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4", T2IParamTypes.ClipGModel);
        }

        public string GetHiDreamClipLModel()
        {
            return RequireClipModel("long_clip_l_hi_dream.safetensors", "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/clip_l_hidream.safetensors", "706fdb88e22e18177b207837c02f4b86a652abca0302821f2bfa24ac6aea4f71", T2IParamTypes.ClipLModel);
        }

        public string GetHiDreamClipGModel()
        {
            return RequireClipModel("long_clip_g_hi_dream.safetensors", "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/clip_g_hidream.safetensors", "3771e70e36450e5199f30bad61a53faae85a2e02606974bcda0a6a573c0519d5", T2IParamTypes.ClipGModel);
        }

        public string GetLlava3Model()
        {
            return RequireClipModel("llava_llama3_fp8_scaled.safetensors", "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors", "2f0c3ad255c282cead3f078753af37d19099cafcfc8265bbbd511f133e7af250", T2IParamTypes.LLaVAModel);
        }

        public string GetLlama31_8b_Model()
        {
            return RequireClipModel("llama_3.1_8b_instruct_fp8_scaled.safetensors", "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors", "9f86897bbeb933ef4fd06297740edb8dd962c94efcd92b373a11460c33765ea6", T2IParamTypes.LLaMAModel);
        }

        public string GetGemma2Model()
        {
            // TODO: Selector param?
            return RequireClipModel("gemma_2_2b_fp16.safetensors", "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/text_encoders/gemma_2_2b_fp16.safetensors", "29761442862f8d064d3f854bb6fabf4379dcff511a7f6ba9405a00bd0f7e2dbd", null);
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
                ["clip_name"] = helpers.GetT5XXLModel(),
                ["type"] = "sd3"
            });
            LoadingClip = [singleClipLoader, 0];
            helpers.DoVaeLoader(UserInput.SourceSession?.User?.Settings?.VAEs?.DefaultSDXLVAE, "stable-diffusion-xl-v1", "sdxl-vae");
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
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX) || model.Metadata?.SpecialFormat == "fp8_scaled" || CurrentCompatClass() is "nvidia-cosmos-predict2" or "chroma" or "chroma-radiance" || CurrentCompatClass().StartsWith("omnigen-")) // TODO: Or AMD?
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
            helpers.DoVaeLoader(null, "nvidia-sana-1600", "sana-dcae");
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
                ["clip_name"] = requireClipModel("t5xxl", T2IParamTypes.T5XXLModel),
                ["type"] = "sd3"
            });
            LoadingClip = [singleClipLoader, 0];
            doVaeLoader(null, "stable-diffusion-xl-v1", "sdxl-vae");
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
        if (info is not null)
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
                List<string> encoders = info.TextEncoders ?? [];
                string[] encoderFiles = encoders.Select(e => requireClipModel(e, encoderParams.GetValueOrDefault(e))).ToArray();
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
                if (info.ClipType is not null)
                {
                    clipInputs["type"] = info.ClipType;
                }
                string clipLoaderNode = CreateNode(loaderType, clipInputs);
                LoadingClip = [clipLoaderNode, 0];
            }
            if (info.BaseModel is "auraflow-v1" or "chroma" or "chroma-radiance")
            {
                string t5Patch = CreateNode("T5TokenizerOptions", new JObject()
                {
                    ["clip"] = LoadingClip,
                    ["min_padding"] = info.BaseModel == "auraflow-v1" ? 768 : 0,
                    ["min_length"] = info.BaseModel == "auraflow-v1" ? 768 : 0
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
                else doVaeLoader(null, CurrentCompatClass(), info.VAE);
            }
        }
        string predType = model.Metadata?.PredictionType == null ? info.PredType.ToString() : model.Metadata?.PredictionType;
        if (!string.IsNullOrWhiteSpace(predType) && LoadingModel is not null)
        {
            if (predType == "sd3")
            {
                if (CurrentCompatClass().StartsWith("flux-1"))
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
                    string samplingNode = CreateNode(info.SigmaShiftNode, new JObject()
                    {
                        ["model"] = LoadingModel,
                        ["shift"] = UserInput.Get(T2IParamTypes.SigmaShift, 3)
                    });
                    LoadingModel = [samplingNode, 0];
                }
            }
            else if (model.Metadata?.PredictionType != null && model.Metadata?.PredictionType != info.PredType.ToString())
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
