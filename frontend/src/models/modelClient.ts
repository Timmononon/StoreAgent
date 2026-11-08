import api, { getPublicError } from '@/utils/api';
import type {
  ModelAssignmentPayload,
  ModelControlPlane,
  ModelProfilePayload,
  ModelRole,
} from '@/types/models';

export async function getModelControlPlane(): Promise<ModelControlPlane> {
  try {
    return (await api.get<ModelControlPlane>('/v1/models')).data;
  } catch (error) {
    throw getPublicError(error);
  }
}

export async function createModelProfile(payload: ModelProfilePayload): Promise<ModelControlPlane> {
  try {
    return (await api.post<ModelControlPlane>('/v1/models', payload)).data;
  } catch (error) {
    throw getPublicError(error);
  }
}

export async function updateModelProfile(
  profileId: string,
  payload: ModelProfilePayload
): Promise<ModelControlPlane> {
  try {
    return (await api.put<ModelControlPlane>(`/v1/models/${profileId}`, payload)).data;
  } catch (error) {
    throw getPublicError(error);
  }
}

export async function deleteModelProfile(profileId: string): Promise<void> {
  try {
    // 删除响应没有控制面快照，由 Store 在成功后重新加载。
    await api.delete(`/v1/models/${profileId}`);
  } catch (error) {
    throw getPublicError(error);
  }
}

export async function assignModelRole(
  role: ModelRole,
  payload: ModelAssignmentPayload
): Promise<ModelControlPlane> {
  try {
    // 返回完整控制面，让调用方同时更新角色分配与目录哈希。
    return (await api.put<ModelControlPlane>(`/v1/models/assignments/${role}`, payload)).data;
  } catch (error) {
    throw getPublicError(error);
  }
}
