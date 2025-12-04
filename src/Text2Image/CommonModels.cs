using System.IO;
using FreneticUtilities.FreneticExtensions;
using SwarmUI.Core;
using SwarmUI.Utils;

namespace SwarmUI.Text2Image;

/// <summary>Special registry of well-known common core Text2Image models that Swarm may make use of.</summary>
public static class CommonModels
{
    /// <summary>Information about an available model.</summary>
    /// <param name="ID">Shorthand lookup ID.</param>
    /// <param name="DisplayName">Human-readable display name.</param>
    /// <param name="URL">Direct download URL.</param>
    /// <param name="Hash">SHA256 raw file hash for the download.</param>
    /// <param name="FolderType">The models subtype this belongs in.</param>
    /// <param name="FileName">The name of the file to save this as when downloading, including folder subpath.</param>
    public record class ModelInfo(string ID, string DisplayName, string URL, string Hash, string FolderType, string FileName)
    {
        /// <summary>Trigger a download of this model.</summary>
        public async Task DownloadNow(Action<long, long, long> updateProgress = null)
        {
            string folder = Program.T2IModelSets[FolderType].FolderPaths[0];
            string path = $"{folder}/{FileName}";
            if (File.Exists(path))
            {
                Logs.Warning($"Attempted re-download of pre-existing model '{FileName}', skipping.");
                return;
            }
            await Utilities.DownloadFile(URL, path, updateProgress, verifyHash: Hash);
        }
    }

    /// <summary>Set of known downloadable models, mapped from their IDs.</summary>
    public static ConcurrentDictionary<string, ModelInfo> Known = [];

    /// <summary>Register a new known model.</summary>
    public static void Register(ModelInfo info)
    {
        if (!info.FileName.EndsWith(".safetensors"))
        {
            throw new InvalidOperationException("May not register a known model that isn't '.safetensors'.");
        }
        if (info.Hash.Length != 64)
        {
            throw new InvalidOperationException($"Hash looks wrong, has {info.Hash.Length} characters, expected 64.");
        }
        if (!info.URL.StartsWith("https://"))
        {
            throw new InvalidOperationException("URL looks wrong.");
        }
        if (!Program.T2IModelSets.ContainsKey(info.FolderType))
        {
            throw new InvalidOperationException($"Folder type '{info.FolderType}' does not exist in set '{Program.T2IModelSets.Keys.JoinString("', '")}'.");
        }
        if (!Known.TryAdd(info.ID, info))
        {
            throw new InvalidOperationException($"Model ID already registered: {info.ID}");
        }
    }

    /// <summary>Internal method to register the core set of known models.</summary>
    public static void RegisterCoreSet()
    {
        //Register(new("", "", "", "", "", ""));

        // Core Reference Models:
        Register(new("sd15", "Stable Diffusion v1.5", "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors", "e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916", "Stable-Diffusion", "OfficialStableDiffusion/v1-5-pruned-emaonly-fp16.safetensors"));
        Register(new("sd21", "Stable Diffusion v2.1", "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors", "dcd690123cfc64383981a31d955694f6acf2072a80537fdb612c8e58ec87a8ac", "Stable-Diffusion", "OfficialStableDiffusion/v2-1_768-ema-pruned.safetensors"));
        Register(new("sdxl1", "Stable Diffusion XL 1.0 (Base)", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b", "Stable-Diffusion", "OfficialStableDiffusion/sd_xl_base_1.0.safetensors"));
        Register(new("sdxl1refiner", "Stable Diffusion XL 1.0 (Refiner)", "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors", "7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f", "Stable-Diffusion", "OfficialStableDiffusion/sd_xl_refiner_1.0.safetensors"));
        Register(new("sd35large", "Stable Diffusion v3.5 Large", "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/sd3.5_large_fp8_scaled.safetensors", "5ad94d6f951556b1ab6b75930fd4effbafaf3130fe9df440e7f2d05a220dd1be", "Stable-Diffusion", "OfficialStableDiffusion/sd3.5_large_fp8_scaled.safetensors"));
        Register(new("fluxschnell", "Flux.1-Schnell", "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors", "ead426278b49030e9da5df862994f25ce94ab2ee4df38b556ddddb3db093bf72", "Stable-Diffusion", "Flux/flux1-schnell-fp8.safetensors"));
        Register(new("fluxdev", "Flux.1-Dev", "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors", "8e91b68084b53a7fc44ed2a3756d821e355ac1a7b6fe29be760c1db532f3d88a", "Stable-Diffusion", "Flux/flux1-dev-fp8.safetensors"));

        // VAEs
        Register(new("sdxl-vae", "Stable Diffusion XL 1.0 VAE", "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors", "235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1", "VAE", "sdxl_vae.safetensors"));
        Register(new("sd35-vae", "Stable Diffusion v3.5 VAE", "https://huggingface.co/mcmonkey/swarm-vaes/resolve/main/sd35_vae.safetensors", "6ad8546282f0f74d6a1184585f1c9fe6f1509f38f284e7c4f7ed578554209859", "VAE", "sd35_vae.safetensors"));
        Register(new("flux-vae", "Flux.1-AE", "https://huggingface.co/mcmonkey/swarm-vaes/resolve/main/flux_ae.safetensors", "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38", "VAE", "Flux/ae.safetensors"));
        Register(new("flux2-vae", "Flux.2-VAE", "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors", "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5", "VAE", "Flux/flux2-vae.safetensors"));
        Register(new("mochi-vae", "Genmo Mochi 1 VAE", "https://huggingface.co/Comfy-Org/mochi_preview_repackaged/resolve/main/split_files/vae/mochi_vae.safetensors", "1be451cec94b911980406169286babc5269e7cf6a94bbbbdf45e8d3f2c961083", "VAE", "mochi_vae.safetensors"));
        Register(new("sana-dcae", "NVIDIA Sana DC-AE", "https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers/resolve/38ebe9b227c30cf6b35f2b7871375e9a28c0ccce/vae/diffusion_pytorch_model.safetensors", "25a1d9ac3b3422160ce8a4b5454ed917f103bb18e30fc1b307dec66375167bb8", "VAE", "sana_dcae_vae.safetensors"));
        Register(new("hunyuan-video-vae", "Hunyuan Video VAE", "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/vae/hunyuan_video_vae_bf16.safetensors", "e8f8553275406d84ccf22e7a47601650d8f98bdb8aa9ccfdd6506b57a9701aed", "VAE", "hunyuan_video_vae_bf16.safetensors"));
        Register(new("cosmos-vae", "Cosmos VAE", "https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/resolve/main/vae/cosmos_cv8x8x8_1.0.safetensors", "e4478fa8629160d16262276e52bdea91ecef636b005a2a29e93a3d7764e0863b", "VAE", "cosmos_cv8x8x8_1.0.safetensors"));
        Register(new("wan21-vae", "Wan 2.1 VAE", "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors", "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b", "VAE", "wan_2.1_vae.safetensors"));
        Register(new("wan22-vae", "Wan 2.2 VAE", "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors", "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156", "VAE", "wan2.2_vae.safetensors"));
        Register(new("ltxv-vae", "LTX-V VAE", "https://huggingface.co/wsbagnsv1/ltxv-13b-0.9.7-dev-GGUF/resolve/c4296d06bab7719ce08e68bfa7a35042898e538b/ltxv-13b-0.9.7-vae-BF16.safetensors", "ee5ddcebc0b92d81b8aed9ee43445b7a4e66df1acf180678c5aa40e82f898dc5", "VAE", "ltxv_vae.safetensors"));
        Register(new("qwen-image-vae", "Qwen Image VAE", "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors", "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f", "VAE", "qwen_image_vae.safetensors"));
        Register(new("hunyuan-image-2_1-vae", "Hunyuan Image 2.1 VAE", "https://huggingface.co/Comfy-Org/HunyuanImage_2.1_ComfyUI/resolve/main/split_files/vae/hunyuan_image_2.1_vae_fp16.safetensors", "f2ae19863609206196b5e3a86bfd94f67bd3866f5042004e3994f07e3c93b2f9", "VAE", "hunyuan_image_2.1_vae_fp16.safetensors"));
        Register(new("hunyuan-image-2_1-refiner-vae", "Hunyuan Image 2.1 Refiner VAE", "https://huggingface.co/Comfy-Org/HunyuanImage_2.1_ComfyUI/resolve/main/split_files/vae/hunyuan_image_refiner_vae_fp16.safetensors", "e1b74e85d61b65e18cc05ca390e387d93cfadf161e737de229ebb800ea3db769", "VAE", "hunyuan_image_2.1_refiner_vae_fp16.safetensors"));
        Register(new("hunyuan-video-1_5-vae", "Hunyuan Video 1.5 VAE", "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/vae/hunyuanvideo15_vae_fp16.safetensors", "e7c3091949c27e2d55ae6d5df917b99dadfebbf308e5a50d0ade0d16c90297ae", "VAE", "HunyuanVideo/hunyuanvideo15_vae_fp16.safetensors"));

        /// text encoders
        Register(new("clip-l", "CLIP-L", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.fp16.safetensors", "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd", "Clip", "clip_l.safetensors"));
        Register(new("clip-g", "CLIP-G", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.fp16.safetensors", "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4", "Clip", "clip_g.safetensors"));
        Register(new("t5xxl", "T5-XXL", "https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/resolve/main/t5xxl_fp8_e4m3fn.safetensors", "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09", "Clip", "t5xxl_enconly.safetensors"));
        Register(new("old-t5xxl", "Old T5-XXL", "https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/resolve/main/text_encoders/oldt5_xxl_fp8_e4m3fn_scaled.safetensors", "1d0dd711ec9866173d4b39e86db3f45e1614a4e3f84919556f854f773352ea81", "Clip", "old_t5xxl_cosmos.safetensors"));
        Register(new("umt5xxl", "UniMax T5-XXL", "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68", "Clip", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"));
        Register(new("pile-t5xxl", "Pile T5-XXL", "https://huggingface.co/fal/AuraFlow-v0.2/resolve/main/text_encoder/model.safetensors", "0a07449cf1141c0ec86e653c00465f6f0d79c6e58a2c60c8bcf4203d0e4ec4f6", "Clip", "pile_t5xl_auraflow.safetensors"));
        Register(new("byt5-small-glyphxl", "ByT5-Small GlyphXL", "https://huggingface.co/Comfy-Org/HunyuanImage_2.1_ComfyUI/resolve/main/split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", "516910bb4c9b225370290e40585d1b0e6c8cd3583690f7eec2f7fb593990fb48", "Clip", "byt5_small_glyphxl_fp16.safetensors"));
        Register(new("qwen-2.5-vl-fp16", "Qwen-2.5-VL", "https://huggingface.co/Comfy-Org/Omnigen2_ComfyUI_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_fp16.safetensors", "ba05dd266ad6a6aa90f7b2936e4e775d801fb233540585b43933647f8bc4fbc3", "Clip", "qwen_2.5_vl_fp16.safetensors"));
        Register(new("qwen-2.5-vl-7b", "Qwen-2.5-VL-7B", "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4", "Clip", "qwen_2.5_vl_7b_fp8_scaled.safetensors"));
        Register(new("clip-l-hidream", "Long CLIP-L", "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/clip_l_hidream.safetensors", "706fdb88e22e18177b207837c02f4b86a652abca0302821f2bfa24ac6aea4f71", "Clip", "long_clip_l_hi_dream.safetensors"));
        Register(new("clip-g-hidream", "Long CLIP-G", "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/clip_g_hidream.safetensors", "3771e70e36450e5199f30bad61a53faae85a2e02606974bcda0a6a573c0519d5", "Clip", "long_clip_g_hi_dream.safetensors"));
        Register(new("llava-llama3", "LLaVA-LLaMA-3", "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors", "2f0c3ad255c282cead3f078753af37d19099cafcfc8265bbbd511f133e7af250", "Clip", "llava_llama3_fp8_scaled.safetensors"));
        Register(new("llama3.1-8b", "LLaMA-3.1-8B", "https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors", "9f86897bbeb933ef4fd06297740edb8dd962c94efcd92b373a11460c33765ea6", "Clip", "llama_3.1_8b_instruct_fp8_scaled.safetensors"));
        Register(new("gemma2-2b", "Gemma-2-2B", "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/text_encoders/gemma_2_2b_fp16.safetensors", "29761442862f8d064d3f854bb6fabf4379dcff511a7f6ba9405a00bd0f7e2dbd", "Clip", "gemma_2_2b_fp16.safetensors"));
        Register(new("qwen_3_4b", "Qwen-3-4B", "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors", "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a", "Clip", "qwen_3_4b.safetensors"));
        Register(new("mistral_3_small_fp8", "Mistrall 3-Small", "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors", "e3467b7d912a234fb929cdf215dc08efdb011810b44bc21081c4234cc75b370e", "Clip", "mistral_3_small_flux2_fp8.safetensors"));
        Register(new("ovis_2.5-5b", "Ovis 2.5-5B", "https://huggingface.co/Comfy-Org/Ovis-Image/resolve/main/split_files/text_encoders/ovis_2.5.safetensors", "f453ee5e7a25cb23cf2adf7aae3e5b405f22097cb67f2cfcca029688cb3f740d", "Clip", "ovis_2.5.safetensors"));

        // Clip Vision
        Register(new("wan21-clipvision-h", "Wan 2.1 ClipVision-H", "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors", "64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161", "ClipVision", "clip_vision_h.safetensors"));
        Register(new("svd-clipvision-h", "SVD ClipVision-H", "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors", "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030", "ClipVision", "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"));
        Register(new("llava-llama3-vision", "LLaVA-LLaMA-3 Vision", "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/clip_vision/llava_llama3_vision.safetensors", "7d0f89bf7860815f3a994b9bdae8ebe3a29c161825d03ca9262cb13b0c973aa6", "ClipVision", "llava_llama3_vision.safetensors"));
        Register(new("sigclip_vision", "SigClip Vision", "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors", "1fee501deabac72f0ed17610307d7131e3e9d1e838d0363aa3c2b97a6e03fb33", "ClipVision", "sigclip_vision_patch14_384.safetensors"));
        Register(new("clip_vision_g", "Clip Vision G", "https://huggingface.co/stabilityai/control-lora/resolve/main/revision/clip_vision_g.safetensors", "9908329b3ead722a693ea400fab1d7c9ec91d6736fd194a94d20d793457f9c2e", "ClipVision", "clip_vision_g.safetensors"));
        Register(new("clip_vision_h", "Clip Vision H", "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors", "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030", "ClipVision", "clip_vision_h.safetensors"));
        Register(new("CLIP-ViT-G", "Clip ViT G", "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors", "657723e09f46a7c3957df651601029f66b1748afb12b419816330f16ed45d64d", "ClipVision", "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"));
        Register(new("CLIP-ViT-H", "Clip Vision H", "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors", "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030", "ClipVision", "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"));
    }
}
