import requests
#"https://www.google.com/url?sa=E&source=gmail&q=http://127.0.0.1:8000/docs"

url = "http://127.0.0.1:8000/upload-audio"
file_path = "IMG_5608.wav"

try:
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "audio/wav")}
        print(f"ファイルを送信中: {file_path}...")
        response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        print("\n--- 解析結果 ---")
        
        top = result['top_result']
        print(f"第1候補: {top['label']} (確信度: {top['prob']})")
        
        print("\n--- 他の候補 ---")
        # enumerateを使って順位（2、3...）を自動で割り当てる
        for i, candidate in enumerate(result['other_candidates'], start=2):
            print(f"第{i}候補: {candidate['label']} ({candidate['prob']})")
    else:
        print(f"エラー: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"予期せぬエラー: {e}")