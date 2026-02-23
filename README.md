# 圍棋定石AI互動平台

一個以 React + Flask 實作的圍棋教學工具，幫助使用者透過與 AI 互動學習定石，培養良好的下棋直覺與判斷能力。

 ## 🚀 體驗入口

- 體驗網站：https://go-joseki-interactive.vercel.app
- ⏳ *由於後端使用免費平台，初次開啟可能需等待伺服器喚醒*
- 本專案後端已部署於 Render，透過 Flask 提供 API 服務（不開放直接存取）
  
## 📽️ 影片介紹

[![觀看介紹影片](https://img.youtube.com/vi/kiUH-brnqGY/0.jpg)](https://www.youtube.com/watch?v=kiUH-brnqGY)

點擊圖片即可觀看完整介紹。

## 📸 平台截圖

<table>
  <tr>
    <td>
      <img src="frontend/src/assets/Goborad.png" alt="平台截圖" width="300"/>
    </td>
    <td>
      <img src="frontend/src/assets/AIexplain.png" alt="平台截圖" width="300"/>
    </td>
  </tr>
  <tr>
    <td align="center">平台棋盤樣式</td>
    <td align="center">AI 即時回饋功能</td>
  </tr>
</table>

## 功能特色

- 📚 題庫導覽：支援題目切換與目錄導覽
- ✏️ 定石填空：可填入後與AI互動，並即時判定對錯
- 🎞️ 棋譜回放：支援「一鍵到底」與「回到起手」
- 🌐 支援部署：可於 Render（後端）與 Vercel（前端）分別部署

## 技術架構

| 元件 | 技術 |
|------|------|
| 前端 | React + Vite |
| 後端 | Flask + Gemini API |
| 部署 | Render（API）+ Vercel（前端） |

## ⚙️ 進階用戶：本地執行方式

如欲自行測試前後端，可參考以下指令：

```bash
# 前端
cd frontend
npm install
npm run dev

# 後端
cd backend
pip install -r requirements.txt
python app.py
```
請在 backend/.env 中設定 GEMINI_API_KEY。注意：本專案不會公開金鑰，請自行註冊。

# 📄 License
本專案採用 MIT License 授權，詳見 LICENSE 檔案。

# 參考資料
- Nihon Ki-in, “楽しい囲碁入門,” [Online]. Available:https://www.nihonkiin.or.jp/teach/lesson/school/syoubu.html 
- 石田芳夫,《初級定石》,  吳仁譯, 世界文物出版社. 
- 大竹英雄,《定石的著想》, 倪炳富譯, 凡異出版社.

# 開發者
陳彥辰（專題設計、題庫設計、前後端整合、觀測系統設計）
> 本專題在開發過程中，有使用 ChatGPT 作為技術協助工具。部分程式碼參考並複製自 AI 回覆內容，但所有內容都經由本人理解、判斷與實測後納入，並根據實際需求進行調整。AI 被視為輔助工具，非完全替代開發工作。
