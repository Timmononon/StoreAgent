import api, { getPublicError } from '@/utils/api';

const ARTIFACT_PATH = /^\/api\/artifacts\/art_[A-Za-z0-9_-]+$/;

export function isFetchableArtifactUri(uri: string | null | undefined): uri is string {
  return typeof uri === 'string' && ARTIFACT_PATH.test(uri);
}

export async function fetchArtifact(uri: string): Promise<Blob> {
  // 下载只接受应用内 Artifact 路径，由 api 实例携带当前认证信息。
  if (!isFetchableArtifactUri(uri)) {
    throw getPublicError({
      code: 'INVALID_REQUEST',
      retryable: false,
      category: 'artifact',
    });
  }
  try {
    // 保留二进制响应，避免下载内容被默认 JSON 或文本转换破坏。
    const response = await api.get<Blob>(uri, { responseType: 'blob' });
    return response.data;
  } catch (error) {
    throw getPublicError(error);
  }
}
