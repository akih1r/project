import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  const formData = await request.formData();
  const file = formData.get("audio") as File; 

  // バックエンドに転送
  // BACKEND_URL: Docker Compose内では "http://backend:8000"
  //              ローカル開発では "http://localhost:8000"（.env.localで設定）
  const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";

  const backendForm = new FormData();
  backendForm.append("file", file, "recording.webm");

  const response = await fetch(`${backendUrl}/upload-audio`, {
    method: "POST",
    body: backendForm,
  });

  const result = await response.json();
  return NextResponse.json(result);
}
