import tgpu from "typegpu"
import type { TgpuRoot } from "typegpu"

let configuredRoot: TgpuRoot | null = null

/** Configure typenet to use an existing TypeGPU root. */
export function configureTypeGPU(root: TgpuRoot): void {
  configuredRoot = root
}

/** Create and configure a TypeGPU root using the current WebGPU adapter. */
export async function initTypeGPU(): Promise<TgpuRoot> {
  const root = await tgpu.init()
  configureTypeGPU(root)
  return root
}

export function getTypeGPURoot(): TgpuRoot {
  if (!configuredRoot) {
    throw new Error(
      "TypeGPU is not configured. Call initTypeGPU() or configureTypeGPU(root) before gpu()."
    )
  }
  return configuredRoot
}

export function clearTypeGPU(root?: TgpuRoot): void {
  if (!root || configuredRoot === root)
    configuredRoot = null
}
