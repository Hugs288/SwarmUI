﻿using FreneticUtilities.FreneticExtensions;
using FreneticUtilities.FreneticToolkit;
using Microsoft.AspNetCore.Builder;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
using SwarmUI.Backends;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using SwarmUI.WebAPI;
using System.IO;
using System.Net.Http;

namespace SwarmUI.Builtin_ComfyUIBackend;

/// <summary>Main class for the ComfyUI Backend extension.</summary>
public class ComfyUIBackendExtension : Extension
{
    /// <summary>Copy of <see cref="Extension.FilePath"/> for ComfyUI.</summary>
    public static string Folder;

    public static PermInfoGroup ComfyPermGroup = new("ComfyUI", "Permissions related to direct interaction with the ComfyUI backend.");

    public static PermInfo PermDirectCalls = Permissions.Register(new("comfy_direct_calls", "ComfyUI Direct Calls", "Allows the user to make direct calls to the ComfyUI backend. Required for most ComfyUI features.", PermissionDefault.POWERUSERS, ComfyPermGroup));
    public static PermInfo PermBackendGenerate = Permissions.Register(new("comfy_backend_generate", "ComfyUI Backend Generate", "Allows the user to generate directly from the ComfyUI backend.", PermissionDefault.POWERUSERS, ComfyPermGroup));
    public static PermInfo PermDynamicCustomWorkflows = Permissions.Register(new("comfy_dynamic_custom_workflows", "ComfyUI Dynamic Custom Workflows", "Allows the user to use dynamic custom workflows via Generate tab parameters.", PermissionDefault.POWERUSERS, ComfyPermGroup));
    public static PermInfo PermStoredCustomWorkflows = Permissions.Register(new("comfy_stored_custom_workflows", "ComfyUI Stored Custom Workflows", "Allows the user to use stored (already saved by a user with direct access) custom workflows via Generate tab parameters.", PermissionDefault.POWERUSERS, ComfyPermGroup));
    public static PermInfo PermReadWorkflows = Permissions.Register(new("comfy_read_workflows", "ComfyUI Read Workflows", "Allows the user read stored workflow data.", PermissionDefault.POWERUSERS, ComfyPermGroup));
    public static PermInfo PermEditWorkflows = Permissions.Register(new("comfy_edit_workflows", "ComfyUI Edit Workflows", "Allows the save, delete, or edit stored workflows.", PermissionDefault.POWERUSERS, ComfyPermGroup));

    public record class ComfyCustomWorkflow(string Name, string Workflow, string Prompt, string CustomParams, string ParamValues, string Image, string Description, bool EnableInSimple);

    /// <summary>All current custom workflow IDs mapped to their data.</summary>
    public static ConcurrentDictionary<string, ComfyCustomWorkflow> CustomWorkflows = new();

    /// <summary>Set of all feature-ids supported by ComfyUI backends.</summary>
    public static HashSet<string> FeaturesSupported = ["comfyui", "refiners", "controlnet", "endstepsearly", "seamless", "video", "variation_seed", "yolov8"];

    /// <summary>Set of feature-ids that were added presumptively during loading and should be removed if the backend turns out to be missing them.</summary>
    public static HashSet<string> FeaturesDiscardIfNotFound = ["variation_seed", "yolov8"];

    /// <summary>Extensible map of ComfyUI Node IDs to supported feature IDs.</summary>
    public static Dictionary<string, string> NodeToFeatureMap = new()
    {
        ["SwarmLoadImageB64"] = "comfy_loadimage_b64",
        ["SwarmSaveImageWS"] = "comfy_saveimage_ws",
        ["SwarmJustLoadTheModelPlease"] = "comfy_just_load_model",
        ["SwarmLatentBlendMasked"] = "comfy_latent_blend_masked",
        ["SwarmKSampler"] = "variation_seed",
        ["AITemplateLoader"] = "aitemplate",
        ["IPAdapter"] = "ipadapter",
        ["IPAdapterApply"] = "ipadapter",
        ["IPAdapterModelLoader"] = "cubiqipadapter",
        ["IPAdapterUnifiedLoader"] = "cubiqipadapterunified",
        ["MiDaS-DepthMapPreprocessor"] = "controlnetpreprocessors",
        ["RIFE VFI"] = "frameinterps",
        ["GIMMVFI_interpolate"] = "frameinterps_gimmvfi",
        ["Sam2Segmentation"] = "sam2",
        ["SwarmYoloDetection"] = "yolov8",
        ["PixArtCheckpointLoader"] = "extramodelspixart",
        ["SanaCheckpointLoader"] = "extramodelssana",
        ["CheckpointLoaderNF4"] = "bnb_nf4",
        ["UnetLoaderGGUF"] = "gguf",
        ["NunchakuFluxDiTLoader"] = "nunchaku",
        ["TensorRTLoader"] = "tensorrt",
        ["TeaCache"] = "teacache",
        ["TeaCacheForVidGen"] = "teacache",
        ["TeaCacheForImgGen"] = "teacache_oldvers",
        ["OverrideCLIPDevice"] = "set_clip_device"
    };

    /// <inheritdoc/>
    public override void OnPreInit()
    {
        Folder = FilePath;
        LoadWorkflowFiles();
        Program.ModelRefreshEvent += Refresh;
        Program.ModelPathsChangedEvent += OnModelPathsChanged;
        ScriptFiles.Add("Assets/comfy_workflow_editor_helper.js");
        StyleSheetFiles.Add("Assets/comfy_workflow_editor.css");
        T2IParamTypes.FakeTypeProviders.Add(DynamicParamGenerator);
        // Temporary: remove old pycache files where we used to have python files, to prevent Comfy boot errors
        Utilities.RemoveBadPycacheFrom($"{FilePath}/ExtraNodes");
        Utilities.RemoveBadPycacheFrom($"{FilePath}/ExtraNodes/SwarmWebHelper");
        T2IAPI.AlwaysTopKeys.Add("comfyworkflowraw");
        T2IAPI.AlwaysTopKeys.Add("comfyworkflowparammetadata");
        if (Directory.Exists($"{FilePath}/DLNodes/ComfyUI_IPAdapter_plus"))
        {
            FeaturesSupported.UnionWith(["ipadapter", "cubiqipadapterunified"]);
            FeaturesDiscardIfNotFound.UnionWith(["ipadapter", "cubiqipadapterunified"]);
        }
        if (Directory.Exists($"{FilePath}/DLNodes/comfyui_controlnet_aux"))
        {
            FeaturesSupported.UnionWith(["controlnetpreprocessors"]);
            FeaturesDiscardIfNotFound.UnionWith(["controlnetpreprocessors"]);
        }
        if (Directory.Exists($"{FilePath}/DLNodes/ComfyUI-Frame-Interpolation"))
        {
            FeaturesSupported.UnionWith(["frameinterps"]);
            FeaturesDiscardIfNotFound.UnionWith(["frameinterps"]);
        }
        if (Directory.Exists($"{FilePath}/DLNodes/ComfyUI-GIMM-VFI"))
        {
            FeaturesSupported.UnionWith(["frameinterps_gimmvfi"]);
            FeaturesDiscardIfNotFound.UnionWith(["frameinterps_gimmvfi"]);
        }
        if (Directory.Exists($"{FilePath}/DLNodes/ComfyUI-segment-anything-2"))
        {
            FeaturesSupported.UnionWith(["sam2"]);
            FeaturesDiscardIfNotFound.UnionWith(["sam2"]);
        }
        if (Directory.Exists($"{FilePath}/DLNodes/ComfyUI_bitsandbytes_NF4"))
        {
            FeaturesSupported.UnionWith(["bnb_nf4"]);
            FeaturesDiscardIfNotFound.UnionWith(["bnb_nf4"]);
        }
        if (Directory.Exists($"{FilePath}/DLNodes/ComfyUI-GGUF"))
        {
            FeaturesSupported.UnionWith(["gguf"]);
            FeaturesDiscardIfNotFound.UnionWith(["gguf"]);
        }
        if (Directory.Exists($"{FilePath}/DLNodes/ComfyUI-TeaCache"))
        {
            FeaturesSupported.UnionWith(["teacache"]);
            FeaturesDiscardIfNotFound.UnionWith(["teacache"]);
        }
        T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.UpscalerModels, InternalListModelsFor("upscale_models", true).Select(u => $"model-{u}///Model: {u}"));
        T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.YoloModels, InternalListModelsFor("yolov8", false));
        T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.GligenModels, InternalListModelsFor("gligen", false));
        T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.StyleModels, InternalListModelsFor("style_models", true));
        SwarmSwarmBackend.OnSwarmBackendAdded += OnSwarmBackendAdded;
    }

    /// <summary>Helper to quickly read a list of model files in a model subfolder, for prepopulating model lists during startup.</summary>
    public static string[] InternalListModelsFor(string subpath, bool createDir)
    {
        string path = Utilities.CombinePathWithAbsolute(Program.ServerSettings.Paths.ActualModelRoot, subpath);
        if (createDir)
        {
            Directory.CreateDirectory(path);
        }
        else if (!Directory.Exists(path))
        {
            return [];
        }
        static bool isModelFile(string f) => T2IModel.LegacyModelExtensions.Contains(f.AfterLast('.')) || T2IModel.NativelySupportedModelExtensions.Contains(f.AfterLast('.'));
        return [.. Directory.EnumerateFiles(path, "*.*", SearchOption.AllDirectories).Where(isModelFile).Select(f => Path.GetRelativePath(path, f))];
    }

    /// <inheritdoc/>
    public override void OnShutdown()
    {
        T2IParamTypes.FakeTypeProviders.Remove(DynamicParamGenerator);
    }

    /// <summary>Forces all currently running comfy backends to restart.</summary>
    public static async Task RestartAllComfyBackends()
    {
        List<Task> tasks = [];
        foreach (ComfyUIAPIAbstractBackend backend in RunningComfyBackends)
        {
            tasks.Add(Program.Backends.ReloadBackend(backend.BackendData));
        }
        await Task.WhenAll(tasks);
    }

    public static T2IParamType FakeRawInputType = new("comfyworkflowraw", "", "", Type: T2IParamDataType.TEXT, ID: "comfyworkflowraw", FeatureFlag: "comfyui", HideFromMetadata: true), // TODO: Setting to toggle metadata
        FakeParameterMetadata = new("comfyworkflowparammetadata", "", "", Type: T2IParamDataType.TEXT, ID: "comfyworkflowparammetadata", FeatureFlag: "comfyui", HideFromMetadata: true);

    public static SingleCacheAsync<string, JObject> ParameterMetadataCacheHelper = new(s => s.ParseToJson());

    public T2IParamType DynamicParamGenerator(string name, T2IParamInput context)
    {
        try
        {
            if (!context.SourceSession?.User?.HasPermission(PermDynamicCustomWorkflows) ?? false)
            {
                return null;
            }
            if (name == "comfyworkflowraw")
            {
                return FakeRawInputType;
            }
            if (name == "comfyworkflowparammetadata")
            {
                return FakeParameterMetadata;
            }
            if (context.TryGetRaw(FakeParameterMetadata, out object paramMetadataObj))
            {
                JObject paramMetadata = ParameterMetadataCacheHelper.GetValue((string)paramMetadataObj);
                if (paramMetadata.TryGetValue(name, out JToken paramTok))
                {
                    T2IParamType type = T2IParamType.FromNet((JObject)paramTok);
                    if (type.Type == T2IParamDataType.INTEGER && type.ViewType == ParamViewType.SEED)
                    {
                        string seedClean(string prior, string newVal)
                        {
                            long parsed = long.Parse(newVal);
                            if (parsed == -1)
                            {
                                int max = (int)type.Max;
                                parsed = Random.Shared.Next(0, max <= 0 ? int.MaxValue : max);
                            }
                            return parsed.ToString();
                        }
                        type = type with { Clean = seedClean };
                    }
                    return type;
                }
                //Logs.Verbose($"Failed to find param metadata for {name} in {paramMetadata.Properties().Select(p => p.Name).JoinString(", ")}");
            }
            if (name.StartsWith("comfyrawworkflowinput") && (context.InternalSet.ValuesInput.ContainsKey("comfyworkflowraw") || context.InternalSet.ValuesInput.ContainsKey("comfyuicustomworkflow")))
            {
                string nameNoPrefix = name.After("comfyrawworkflowinput");
                T2IParamDataType type = FakeRawInputType.Type;
                ParamViewType numberType = ParamViewType.BIG;
                Func<string, string, string> cleaner = null;
                if (nameNoPrefix.StartsWith("seed"))
                {
                    type = T2IParamDataType.INTEGER;
                    numberType = ParamViewType.SEED;
                    nameNoPrefix = nameNoPrefix.After("seed");
                    string seedClean(string prior, string newVal)
                    {
                        long parsed = long.Parse(newVal);
                        if (parsed == -1)
                        {
                            parsed = Random.Shared.Next(0, int.MaxValue);
                        }
                        return parsed.ToString();
                    }
                    cleaner = seedClean;
                }
                else
                {
                    foreach (T2IParamDataType possible in Enum.GetValues<T2IParamDataType>())
                    {
                        string typeId = possible.ToString().ToLowerFast();
                        if (nameNoPrefix.StartsWith(typeId))
                        {
                            nameNoPrefix = nameNoPrefix.After(typeId);
                            type = possible;
                            break;
                        }
                    }
                }
                T2IParamType resType = FakeRawInputType with { Name = nameNoPrefix, ID = name, HideFromMetadata = false, Type = type, ViewType = numberType, Clean = cleaner };
                if (type == T2IParamDataType.MODEL)
                {
                    static string cleanup(string _, string val)
                    {
                        val = val.Replace('\\', '/');
                        while (val.Contains("//"))
                        {
                            val = val.Replace("//", "/");
                        }
                        val = val.Replace('/', Path.DirectorySeparatorChar);
                        return val;
                    }
                    resType = resType with { Clean = cleanup };
                }
                return resType;
            }
        }
        catch (Exception e)
        {
            Logs.Error($"Error generating dynamic Comfy param {name}: {e}");
        }
        return null;
    }

    public static IEnumerable<ComfyUIAPIAbstractBackend> RunningComfyBackends => Program.Backends.RunningBackendsOfType<ComfyUIAPIAbstractBackend>();

    public static string[] ExampleWorkflowNames;

    public void LoadWorkflowFiles()
    {
        CustomWorkflows.Clear();
        Directory.CreateDirectory($"{FilePath}/CustomWorkflows");
        Directory.CreateDirectory($"{FilePath}/CustomWorkflows/Examples");
        string[] getCustomFlows(string path) => [.. Directory.EnumerateFiles($"{FilePath}/{path}", "*.*", new EnumerationOptions() { RecurseSubdirectories = true }).Select(f => f.Replace('\\', '/').After($"/{path}/")).Order()];
        ExampleWorkflowNames = getCustomFlows("ExampleWorkflows");
        string[] customFlows = getCustomFlows("CustomWorkflows");
        bool anyCopied = false;
        foreach (string workflow in ExampleWorkflowNames.Where(f => f.EndsWith(".json")))
        {
            if (!customFlows.Contains($"Examples/{workflow}") && !customFlows.Contains($"Examples/{workflow}.deleted"))
            {
                File.Copy($"{FilePath}/ExampleWorkflows/{workflow}", $"{FilePath}/CustomWorkflows/Examples/{workflow}");
                anyCopied = true;
            }
        }
        if (anyCopied)
        {
            customFlows = getCustomFlows("CustomWorkflows");
        }
        foreach (string workflow in customFlows.Where(f => f.EndsWith(".json")))
        {
            CustomWorkflows.TryAdd(workflow.BeforeLast('.'), null);
        }
    }

    public static ComfyCustomWorkflow GetWorkflowByName(string name)
    {
        if (!CustomWorkflows.TryGetValue(name, out ComfyCustomWorkflow workflow))
        {
            return null;
        }
        if (workflow is not null)
        {
            return workflow;
        }
        string path = $"{Folder}/CustomWorkflows/{name}.json";
        if (!File.Exists(path))
        {
            CustomWorkflows.TryRemove(name, out _);
            return null;
        }
        JObject json = File.ReadAllText(path).ParseToJson();
        string getStringFor(string key)
        {
            if (!json.TryGetValue(key, out JToken data))
            {
                return null;
            }
            if (data.Type == JTokenType.String)
            {
                return data.ToString();
            }
            return data.ToString(Formatting.None);
        }
        string workflowData = getStringFor("workflow");
        string prompt = getStringFor("prompt");
        string customParams = getStringFor("custom_params");
        string paramValues = getStringFor("param_values");
        string image = getStringFor("image") ?? "/imgs/model_placeholder.jpg";
        string description = getStringFor("description");
        bool enableInSimple = json.TryGetValue("enable_in_simple", out JToken enableInSimpleTok) && enableInSimpleTok.ToObject<bool>();
        workflow = new(name, workflowData, prompt, customParams, paramValues, image, description, enableInSimple);
        CustomWorkflows[name] = workflow;
        return workflow;
    }

    public void Refresh()
    {
        List<Task> tasks = [];
        try
        {
            ComfyUIRedirectHelper.ObjectInfoReadCacher.ForceExpire();
            LoadWorkflowFiles();
            foreach (ComfyUIAPIAbstractBackend backend in RunningComfyBackends.ToArray())
            {
                tasks.Add(backend.LoadValueSet(5));
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"Error refreshing ComfyUI: {ex.ReadableString()}");
        }
        if (!tasks.Any())
        {
            return;
        }
        try
        {
            using CancellationTokenSource cancel = Utilities.TimedCancel(TimeSpan.FromMinutes(0.5));
            Task.WaitAll([.. tasks], cancel.Token);
        }
        catch (Exception ex)
        {
            Logs.Debug("ComfyUI refresh failed, will retry in background");
            Logs.Verbose($"Error refreshing ComfyUI: {ex.ReadableString()}");
            Utilities.RunCheckedTask(() =>
            {
                using CancellationTokenSource cancel = Utilities.TimedCancel(TimeSpan.FromMinutes(5));
                Task.WaitAll([.. tasks], cancel.Token);
            }, "refreshing ComfyUI");
        }
    }

    public void OnModelPathsChanged()
    {
        ComfyUISelfStartBackend.IsComfyModelFileEmitted = false;
        foreach (ComfyUISelfStartBackend backend in Program.Backends.RunningBackendsOfType<ComfyUISelfStartBackend>())
        {
            if (backend.IsEnabled)
            {
                Program.Backends.ReloadBackend(backend.BackendData).Wait(Program.GlobalProgramCancel);
            }
        }
    }

    public static async Task RunArbitraryWorkflowOnFirstBackend(string workflow, Action<object> takeRawOutput, bool allowRemote = true)
    {
        ComfyUIAPIAbstractBackend backend = RunningComfyBackends.FirstOrDefault(b => allowRemote || b is ComfyUISelfStartBackend) ?? throw new SwarmUserErrorException("No available ComfyUI Backend to run this operation");
        await backend.AwaitJobLive(workflow, "0", takeRawOutput, new(null), Program.GlobalProgramCancel);
    }

    public static void OnSwarmBackendAdded(SwarmSwarmBackend backend)
    {
        // TODO: Multi-layered forwarding? (Swarm connects to Swarm connects to Comfy)
        if (!backend.LinkedRemoteBackendType.StartsWith("comfyui_"))
        {
            return;
        }
        Utilities.RunCheckedTask(async () =>
        {
            HttpRequestMessage getReq = new(HttpMethod.Get, $"{backend.Address}/ComfyBackendDirect/object_info");
            backend.RequestAdapter()?.Invoke(getReq);
            getReq.Headers.Add("X-Swarm-Backend-ID", $"{backend.LinkedRemoteBackendID}");
            HttpResponseMessage resp = await SwarmSwarmBackend.HttpClient.SendAsync(getReq, Program.GlobalProgramCancel);
            JObject rawObjectInfo = (await resp.Content.ReadAsStringAsync()).ParseToJson();
            AssignValuesFromRaw(rawObjectInfo);
        });
    }

    public static LockObject ValueAssignmentLocker = new();

    /// <summary>Add handlers here to do additional parsing of RawObjectInfo data.</summary>
    public static List<Action<JObject>> RawObjectInfoParsers = [];

    public static void AssignValuesFromRaw(JObject rawObjectInfo)
    {
        lock (ValueAssignmentLocker)
        {
            if (rawObjectInfo.TryGetValue("UpscaleModelLoader", out JToken modelLoader))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.UpscalerModels, modelLoader["input"]["required"]["model_name"][0].Select(u => $"model-{u}///Model: {u}"));
            }
            if (rawObjectInfo.TryGetValue("SwarmKSampler", out JToken swarmksampler))
            {
                string[] dropped = [.. T2IParamTypes.Samplers.Select(s => s.Before("///")).Except([.. swarmksampler["input"]["required"]["sampler_name"][0].Select(u => $"{u}")])];
                if (dropped.Any())
                {
                    Logs.Warning($"Samplers are listed, but not included in SwarmKSampler internal list: {dropped.JoinString(", ")}");
                }
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.Samplers, swarmksampler["input"]["required"]["sampler_name"][0].Select(u => $"{u}///{u} (New)"));
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.Schedulers, swarmksampler["input"]["required"]["scheduler"][0].Select(u => $"{u}///{u} (New)"));
            }
            if (rawObjectInfo.TryGetValue("KSampler", out JToken ksampler))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.Samplers, ksampler["input"]["required"]["sampler_name"][0].Select(u => $"{u}///{u} (New in KS)"));
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.Schedulers, ksampler["input"]["required"]["scheduler"][0].Select(u => $"{u}///{u} (New in KS)"));
            }
            if (rawObjectInfo.TryGetValue("IPAdapterUnifiedLoader", out JToken ipadapterCubiqUnified))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.IPAdapterModels, ipadapterCubiqUnified["input"]["required"]["preset"][0].Select(m => $"{m}"));
            }
            else if (rawObjectInfo.TryGetValue("IPAdapter", out JToken ipadapter) && (ipadapter["input"]["required"] as JObject).TryGetValue("model_name", out JToken ipAdapterModelName))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.IPAdapterModels, ipAdapterModelName[0].Select(m => $"{m}"));
            }
            if (rawObjectInfo.TryGetValue("IPAdapterModelLoader", out JToken ipadapterCubiq))
            {
                HashSet<string> native = ["ip-adapter-faceid-portrait-v11_sd15.bin", "ip-adapter-faceid-portrait_sdxl.bin", "ip-adapter-faceid-portrait_sdxl_unnorm.bin", "ip-adapter-faceid-plusv2_sd15.bin", "ip-adapter-faceid-plusv2_sdxl.bin", "ip-adapter-faceid-plus_sd15.bin", "ip-adapter-faceid_sd15.bin", "ip-adapter-faceid_sdxl.bin", "full_face_sd15.safetensors", "ip-adapter-plus-face_sd15.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors", "ip-adapter-plus_sd15.safetensors", "ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter_sd15_vit-G.safetensors", "ip-adapter_sdxl.safetensors", "ip-adapter_sd15.safetensors", "ip-adapter_sdxl_vit-h.safetensors", "sd15_light_v11.bin"];
                string[] models = [.. ipadapterCubiq["input"]["required"]["ipadapter_file"][0].Select(m => $"{m}").Where(m => !native.Contains(m))];
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.IPAdapterModels, models.Select(m => $"file:{m}///Model File: {m}"));
            }
            if (rawObjectInfo.TryGetValue("IPAdapter", out JToken ipadapter2) && (ipadapter2["input"]["required"] as JObject).TryGetValue("weight_type", out JToken ipAdapterWeightType))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.IPAdapterWeightTypes, ipAdapterWeightType[0].Select(m => $"{m}///{m} (New)"));
            }
            if (rawObjectInfo.TryGetValue("IPAdapterUnifiedLoaderFaceID", out JToken ipadapterCubiqUnifiedFace))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.IPAdapterModels, ipadapterCubiqUnifiedFace["input"]["required"]["preset"][0].Select(m => $"{m}"));
            }
            if (rawObjectInfo.TryGetValue("GLIGENLoader", out JToken gligenLoader))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.GligenModels, gligenLoader["input"]["required"]["gligen_name"][0].Select(m => $"{m}"));
            }
            if (rawObjectInfo.TryGetValue("StyleModelLoader", out JToken styleModelLoader))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.StyleModels, styleModelLoader["input"]["required"]["style_model_name"][0].Select(m => $"{m}"));
            }
            if (rawObjectInfo.TryGetValue("SwarmYoloDetection", out JToken yoloDetection))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.YoloModels, yoloDetection["input"]["required"]["model_name"][0].Select(m => $"{m}"));
            }
            if (rawObjectInfo.TryGetValue("SetUnionControlNetType", out JToken unionCtrlNet))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.ControlnetUnionTypes, unionCtrlNet["input"]["required"]["type"][0].Select(m => $"{m}///{m} (New)"));
            }
            if (rawObjectInfo.TryGetValue("OverrideCLIPDevice", out JToken overrideClipDevice))
            {
                T2IParamTypes.ConcatDropdownValsClean(ref T2IParamTypes.SetClipDevices, overrideClipDevice["input"]["required"]["device"][0].Select(m => $"{m}"));
            }
            if (rawObjectInfo.TryGetValue("Sam2AutoSegmentation", out JToken nodeData))
            {
                foreach (string size in new string[] { "base_plus", "large", "small" })
                {
                    T2IParamTypes.ControlNetPreprocessors[$"Segment Anything 2 Global Autosegment {size}"] = new JObject()
                    {
                        ["swarm_custom"] = true,
                        ["output"] = "SWARM:NODE_1,1",
                        ["nodes"] = new JArray()
                        {
                            new JObject()
                            {
                                ["class_type"] = "DownloadAndLoadSAM2Model",
                                ["inputs"] = new JObject()
                                {
                                    ["model"] = $"sam2_hiera_{size}.safetensors",
                                    ["segmentor"] = "automaskgenerator",
                                    ["device"] = "cuda", // TODO: This should really be decided by the python, not by swarm's workflow generator - the python knows what the GPU supports, swarm does not
                                    ["precision"] = "bf16"
                                }
                            },
                            new JObject()
                            {
                                ["class_type"] = "Sam2AutoSegmentation",
                                ["node_data"] = nodeData,
                                ["inputs"] = new JObject()
                                {
                                    ["sam2_model"] = "SWARM:NODE_0",
                                    ["image"] = "SWARM:INPUT_0"
                                }
                            }
                        }
                    };
                }
            }
            foreach ((string key, JToken data) in rawObjectInfo)
            {
                if (data["category"].ToString() == "image/preprocessors")
                {
                    T2IParamTypes.ControlNetPreprocessors[key] = data;
                }
                else if (key.EndsWith("Preprocessor"))
                {
                    T2IParamTypes.ControlNetPreprocessors[key] = data;
                }
                if (NodeToFeatureMap.TryGetValue(key, out string featureId))
                {
                    FeaturesSupported.Add(featureId);
                    FeaturesDiscardIfNotFound.Remove(featureId);
                }
            }
            foreach (string feature in FeaturesDiscardIfNotFound)
            {
                FeaturesSupported.Remove(feature);
            }
            foreach (Action<JObject> parser in RawObjectInfoParsers)
            {
                try
                {
                    parser(rawObjectInfo);
                }
                catch (Exception ex)
                {
                    Logs.Error($"Error while running extension parsing on raw object info: {ex.ReadableString()}");
                }
            }
        }
    }

    public static T2IRegisteredParam<string> WorkflowParam, CustomWorkflowParam, SamplerParam, SchedulerParam, RefinerSamplerParam, RefinerSchedulerParam, RefinerUpscaleMethod, UseIPAdapterForRevision, IPAdapterWeightType, VideoPreviewType, VideoFrameInterpolationMethod, Text2VideoFrameInterpolationMethod, GligenModel, YoloModelInternal, PreferredDType, UseStyleModel, TeaCacheMode, EasyCacheMode, SetClipDevice;

    public static T2IRegisteredParam<bool> AITemplateParam, DebugRegionalPrompting, ShiftedLatentAverageInit, UseCfgZeroStar, UseTCFG;

    public static T2IRegisteredParam<double> IPAdapterWeight, IPAdapterStart, IPAdapterEnd, SelfAttentionGuidanceScale, SelfAttentionGuidanceSigmaBlur, PerturbedAttentionGuidanceScale, StyleModelMergeStrength, StyleModelApplyStart, StyleModelMultiplyStrength, RescaleCFGMultiplier, TeaCacheThreshold, TeaCacheStart, NunchakuCacheThreshold, EasyCacheThreshold, EasyCacheStart, EasyCacheEnd, RenormCFG;

    public static T2IRegisteredParam<int> RefinerHyperTile, VideoFrameInterpolationMultiplier, Text2VideoFrameInterpolationMultiplier;

    public static T2IRegisteredParam<string>[] ControlNetPreprocessorParams = new T2IRegisteredParam<string>[3], ControlNetUnionTypeParams = new T2IRegisteredParam<string>[3];

    public static List<string> UpscalerModels = ["pixel-lanczos///Pixel: Lanczos (cheap + high quality)", "pixel-bicubic///Pixel: Bicubic (Basic)", "pixel-area///Pixel: Area", "pixel-bilinear///Pixel: Bilinear", "pixel-nearest-exact///Pixel: Nearest-Exact (Pixel art)", "latent-bislerp///Latent: Bislerp", "latent-bicubic///Latent: Bicubic", "latent-area///Latent: Area", "latent-bilinear///Latent: Bilinear", "latent-nearest-exact///Latent: Nearest-Exact"],
        Samplers =
        [
            // K-Diffusion
            "euler///Euler", "euler_ancestral///Euler Ancestral (Randomizing)", "heun///Heun (2x Slow)", "heunpp2///Heun++ 2 (2x Slow)", "dpm_2///DPM-2 (Diffusion Probabilistic Model) (2x Slow)", "dpm_2_ancestral///DPM-2 Ancestral (2x Slow)",
            "lms///LMS (Linear Multi-Step)", "dpm_fast///DPM Fast (DPM without the DPM2 slowdown)", "dpm_adaptive///DPM Adaptive (Dynamic Steps)",
            "dpmpp_2s_ancestral///DPM++ 2S Ancestral (2nd Order Single-Step) (2x Slow)", "dpmpp_sde///DPM++ SDE (Stochastic / randomizing) (2x Slow)", "dpmpp_sde_gpu///DPM++ SDE, GPU Seeded (2x Slow)",
            "dpmpp_2m///DPM++ 2M (2nd Order Multi-Step)", "dpmpp_2m_sde///DPM++ 2M SDE", "dpmpp_2m_sde_gpu///DPM++ 2M SDE, GPU Seeded", "dpmpp_3m_sde///DPM++ 3M SDE (3rd Order Multi-Step)", "dpmpp_3m_sde_gpu///DPM++ 3M SDE, GPU Seeded",
            "ddim///DDIM (Denoising Diffusion Implicit Models) (Identical to Euler)", "ddpm///DDPM (Denoising Diffusion Probabilistic Models)",
            // Unique tack-ons
             "lcm///LCM (for LCM models)", "uni_pc///UniPC (Unified Predictor-Corrector)", "uni_pc_bh2///UniPC BH2", "res_multistep///Res MultiStep (for Cosmos)", "res_multistep_ancestral///Res MultiStep Ancestral (randomizing, for Cosmos)",
            "ipndm///iPNDM (Improved Pseudo-Numerical methods for Diffusion Models)", "ipndm_v///iPNDM-V (Variable-Step)", "deis///DEIS (Diffusion Exponential Integrator Sampler)", "gradient_estimation///Gradient Estimation (Improving from Optimization Perspective)",
            "er_sde///ER-SDE-Solver (used with AlignYourSteps schedule)", "seeds_2///SEEDS 2 (Exponential SDE Solvers, variant of DPM++ SDE)", "seeds_3///SEEDS 3", "sa_solver///SA-Solver (Stochastic Adams)", "sa_solver_pece///SA-Solver PECE",
            // CFG++ variants
            "euler_cfg_pp///Euler CFG++ (Manifold-constrained CFG)", "euler_ancestral_cfg_pp///Euler Ancestral CFG++", "dpmpp_2m_cfg_pp///DPM++ 2M CFG++", "dpmpp_2s_ancestral_cfg_pp///DPM++ 2S Ancestral CFG++ (2x Slow)", "res_multistep_cfg_pp///Res MultiStep CFG++", "res_multistep_ancestral_cfg_pp///Res MultiStep Ancestral CFG++", "gradient_estimation_cfg_pp///Gradient Estimation CFG++"
        ],
        Schedulers = ["normal///Normal", "karras///Karras", "exponential///Exponential", "simple///Simple", "ddim_uniform///DDIM Uniform", "sgm_uniform///SGM Uniform", "turbo///Turbo (for turbo models, max 10 steps)", "align_your_steps///Align Your Steps (NVIDIA, rec. 10 steps)", "beta///Beta", "linear_quadratic///Linear Quadratic (Mochi)", "ltxv///LTX-Video", "ltxv-image///LTXV-Image", "kl_optimal///KL Optimal (Nvidia AYS)"];

    public static List<string> IPAdapterModels = ["None"], IPAdapterWeightTypes = ["standard", "prompt is more important", "style transfer"];

    public static List<string> GligenModels = ["None"], YoloModels = [], StyleModels = ["None"], SetClipDevices = ["cpu"];

    public static List<string> ControlnetUnionTypes = ["auto", "openpose", "depth", "hed/pidi/scribble/ted", "canny/lineart/anime_lineart/mlsd", "normal", "segment", "tile", "repaint"];

    public static ConcurrentDictionary<string, JToken> ControlNetPreprocessors = new() { ["None"] = null };

    public static T2IParamGroup ComfyGroup, ComfyAdvancedGroup;
    /// <inheritdoc/>
    public override void OnInit()
    {
        T2IParamTypes.RegisterComfyParams();
        Program.Backends.RegisterBackendType<ComfyUIAPIBackend>("comfyui_api", "ComfyUI API By URL", "A backend powered by a pre-existing installation of ComfyUI, referenced via API base URL.", true);
        Program.Backends.RegisterBackendType<ComfyUISelfStartBackend>("comfyui_selfstart", "ComfyUI Self-Starting", "A backend powered by a pre-existing installation of the ComfyUI, automatically launched and managed by this UI server.", isStandard: true);
        ComfyUIWebAPI.Register();
    }

    public override void OnPreLaunch()
    {
        WebServer.WebApp.Map("/ComfyBackendDirect/{*Path}", ComfyUIRedirectHelper.ComfyBackendDirectHandler);
    }

    public record struct ComfyBackendData(HttpClient Client, string APIAddress, string WebAddress, AbstractT2IBackend Backend);

    public static IEnumerable<ComfyBackendData> ComfyBackendsDirect()
    {
        foreach (ComfyUIAPIAbstractBackend backend in RunningComfyBackends)
        {
            yield return new(ComfyUIAPIAbstractBackend.HttpClient, backend.APIAddress, backend.WebAddress, backend);
        }
        foreach (SwarmSwarmBackend swarmBackend in Program.Backends.RunningBackendsOfType<SwarmSwarmBackend>().Where(b => b.LinkedRemoteBackendType is not null && b.LinkedRemoteBackendType.StartsWith("comfyui_")))
        {
            string addr = $"{swarmBackend.Address}/ComfyBackendDirect";
            yield return new(SwarmSwarmBackend.HttpClient, addr, addr, swarmBackend);
        }
    }
}