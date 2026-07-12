import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { isMac } from '@/helpers/basic';

const prisma = new PrismaClient();

const clone = (obj: any) => JSON.parse(JSON.stringify(obj));

/**
 * Multi-stage jobs: the Krea 2 DRaFT reward stage (process[1+]) is derived
 * from the SFT stage (process[0]). The form only edits reward-specific
 * settings, so on every save we re-sync the shared fields (model, network,
 * folders, trigger) from process[0] and keep step counts cumulative -- the
 * DRaFT stage resumes the SFT checkpoint from the same training folder.
 */
const syncDraftStages = (jobConfig: any) => {
  const processes = jobConfig?.config?.process;
  if (!Array.isArray(processes) || processes.length < 2) return;
  const p0 = processes[0];
  for (let i = 1; i < processes.length; i++) {
    const p = processes[i];
    if (p?.type !== 'krea2_draft_trainer') continue;

    p.training_folder = p0.training_folder;
    p.sqlite_db_path = p0.sqlite_db_path;
    p.device = p0.device;
    p.trigger_word = p0.trigger_word;
    p.performance_log_every = p0.performance_log_every;
    p.model = clone(p0.model);
    p.logging = clone(p0.logging);
    p.sample = clone(p0.sample);
    // resume happens via the shared training folder; no pretrained path
    p.network = clone(p0.network);
    delete p.network.pretrained_lora_path;
    p.save = { ...clone(p0.save), save_every: Math.min(p0.save?.save_every ?? 20, 20) };
    // the DRaFT stage samples its own images from prompts; no datasets
    delete p.datasets;

    const draft = p.draft || {};
    const reward = draft.reward || {};
    const rewardSteps = Number(draft.num_reward_steps ?? 60);
    p.draft = {
      steps: 12,
      draft_k: 1,
      guidance_scale: 4.5,
      width: 512,
      height: 512,
      lv_samples: 0,
      high_noise_shift: 0.5,
      seed: 42,
      checkpoint_vae: true,
      train_modules: 'qkvo',
      save_images_every: 10,
      prompts: null,
      prompts_path: null,
      ...draft,
      num_reward_steps: rewardSteps,
      reward: {
        reference_images: p0.datasets?.[0]?.folder_path ?? null,
        face_weight: 1.0,
        body_weight: 0.5,
        // detector runs on CPU: it is non-differentiable and the ORT CUDA
        // arena wastes VRAM the sampling graph needs
        face: { target_similarity: 0.45, providers: ['CPUExecutionProvider'] },
        body: { loss_tier: 'somax', sam3d_hf_repo: 'facebook/sam-3d-body-vith' },
        ...reward,
      },
    };

    // steps are cumulative: the stage resumes at p0's final step
    p.train = {
      ...clone(p0.train),
      steps: Number(p0.train?.steps ?? 0) + rewardSteps,
      optimizer: 'adamw',
      optimizer_params: { weight_decay: 1e-4 },
      lr: 0.0001,
      max_grad_norm: 1.0,
      skip_first_sample: true,
      cache_text_embeddings: false,
      unload_text_encoder: false,
    };
  }
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');
  const only_active = searchParams.get('only_active');

  try {
    if (id) {
      const job = await prisma.job.findUnique({
        where: { id },
      });
      return NextResponse.json(job);
    }
    if (job_ref) {
      const job = await prisma.job.findFirst({
        where: { job_ref },
        orderBy: { updated_at: 'desc' },
      });
      return NextResponse.json(job);
    }

    const where: any = {};
    if (job_type) {
      where.job_type = job_type;
    }
    if (only_active === 'true') {
      where.status = { in: ['running', 'queued', 'stopping'] };
    }

    const jobs = await prisma.job.findMany({
      where,
      orderBy: { created_at: 'desc' },
    });
    return NextResponse.json({ jobs: jobs });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id, name, job_config } = body;
    let gpu_ids: string = body.gpu_ids;

    syncDraftStages(job_config);

    if (isMac()) {
      gpu_ids = "mps";
    }

    const extra: any = {};
    if ("job_ref" in body) {
      extra["job_ref"] = body.job_ref;
    }

    if ("job_type" in body) {
      extra["job_type"] = body.job_type;
    }

    if (id) {
      // Update existing training
      const training = await prisma.job.update({
        where: { id },
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          ...extra,
        },
      });
      return NextResponse.json(training);
    } else {
      // find the highest queue position and add 1000
      const highestQueuePosition = await prisma.job.aggregate({
        _max: {
          queue_position: true,
        },
      });
      const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

      // Create new training
      const training = await prisma.job.create({
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          queue_position: newQueuePosition,
          ...extra,
        },
      });
      return NextResponse.json(training);
    }
  } catch (error: any) {
    if (error.code === 'P2002') {
      // Handle unique constraint violation, 409=Conflict
      return NextResponse.json({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error(error);
    // Handle other errors
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
