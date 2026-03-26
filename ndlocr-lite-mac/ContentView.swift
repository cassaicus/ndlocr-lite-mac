import SwiftUI
import onnxruntime_objc
import AppKit
internal import UniformTypeIdentifiers

// アプリケーションのメイン画面を定義する構造体。
struct ContentView: View {
    // ユーザーが選択した画像を保持する状態変数。nilの場合は未選択状態。
    @State private var selectedImage: NSImage?
    // OCRによって検出・認識された行データの配列。画面のリスト表示や矩形描画に使用される。
    @State private var ocrLines: [OCRLine] = []
    // OCR処理が実行中かどうかを示すフラグ。UIの無効化やインジケータ表示に使用する。
    @State private var isProcessing = false
    // 処理中にエラーが発生した場合のメッセージを保持する。
    @State private var errorMessage: String?
    
    var body: some View {
        // 画面を左右に分割。左側に画像表示、右側に認識結果を表示する。
        HStack(spacing: 0) {
            // 左側：画像表示エリアと各種操作ボタンを配置。
            // Left Side: Image display and controls
            VStack {
                // 画像が選択されている場合の表示処理。
                if let image = selectedImage {
                    ZStack {
                        // 選択された画像を、アスペクト比を維持しつつ画面にフィットさせて表示。
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFit()
                            .background(Color.black.opacity(0.1))
                            .cornerRadius(12)
                            // 画像の上に、検出されたテキスト行の矩形（バウンディングボックス）を重ねて表示。
                            .overlay(
                                BoundingBoxOverlay(image: image, ocrLines: ocrLines)
                            )
                    }
                    .padding()
                } else {
                    // 画像が選択されていない時のプレースホルダー表示。
                    VStack {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 80))
                            .foregroundColor(.secondary)
                        Text("画像を選択してください")
                            .font(.headline)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                
                // 下部のボタンエリア。画像の選択とOCRの実行を行う。
                HStack(spacing: 16) {
                    // システムのファイル選択ダイアログを開くボタン。
                    Button(action: selectImage) {
                        Label("画像を選択", systemImage: "plus.circle.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.blue)
                    .controlSize(.large)
                    
                    // 選択された画像に対してOCR処理を開始するボタン。
                    Button(action: runOCR) {
                        if isProcessing {
                            // 処理中はボタン内にインジケータを表示して、動作中であることを伝える。
                            ProgressView()
                                .controlSize(.small)
                                .frame(maxWidth: .infinity)
                        } else {
                            Label("OCR実行", systemImage: "sparkles")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.purple)
                    .controlSize(.large)
                    // 処理中や画像未選択時は、二重実行やエラーを防ぐためにボタンを無効化。
                    .disabled(selectedImage == nil || isProcessing)
                }
                .padding()
                .background(Color(NSColor.windowBackgroundColor))
            }
            .frame(minWidth: 400)
            
            // 左右の領域を分ける境界線。
            Divider()
            
            // 右側：OCRによるテキスト認識結果を一覧表示する。
            // Right Side: OCR results
            VStack(alignment: .leading, spacing: 0) {
                // セクションの見出し。
                Text("OCR結果")
                    .font(.headline)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.secondary.opacity(0.1))
                
                // 認識された各行のテキストとスコアをスクロール可能なリストで表示。
                List(ocrLines) { line in
                    VStack(alignment: .leading, spacing: 4) {
                        // 認識された文字列。
                        Text(line.text)
                            .font(.body)
                        // 検出の信頼度スコア。
                        Text("Score: \(String(format: "%.3f", line.score))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 4)
                }
                .listStyle(InsetListStyle())
            }
            .frame(width: 300)
        }
        // ウィンドウの最小サイズを設定。
        .frame(minWidth: 800, minHeight: 600)
        // エラー発生時にポップアップアラートを表示する。
        .alert("エラー", isPresented: Binding<Bool>(
            get: { errorMessage != nil },
            set: { if !$0 { errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { }
        } message: {
            if let msg = errorMessage {
                Text(msg)
            }
        }
    }
    
    // 画像を選択するためのOS標準ファイルオープンダイアログを表示。
    private func selectImage() {
        let panel = NSOpenPanel()
        // 一度に選択できるのは1つの画像のみに制限。
        panel.allowsMultipleSelection = false
        // フォルダの選択は不可。
        panel.canChooseDirectories = false
        // 画像ファイル形式のみを選択可能にする。
        panel.allowedContentTypes = [.image]
        
        // ダイアログを表示し、「開く」が押された場合に処理を行う。
        if panel.runModal() == .OK {
            if let url = panel.url {
                // ファイルURLから画像を読み込み。
                selectedImage = NSImage(contentsOf: url)
                // 新しい画像が選ばれたので、以前の認識結果をクリアする。
                ocrLines = []
            }
        }
    }
    
    // 選択された画像に対してOCR処理を実行する。
    private func runOCR() {
        // 画像が選択されていない場合は何もしない（通常はボタンが無効化されている）。
        guard let image = selectedImage else { return }
        
        // 処理中フラグを立ててUIを更新。
        isProcessing = true
        // 前回の結果をクリア。
        ocrLines = []
        
        // 非同期タスクでOCRエンジンを呼び出す。
        Task {
            do {
                // OCRエンジン（モデル読み込み含む）の初期化。
                let runner = try OCRRunner()
                // 重い推論処理をバックグラウンドスレッドで実行してUIのフリーズを防ぐ。
                let results = try await Task.detached(priority: .userInitiated) {
                    try runner.process(image: image)
                }.value
                
                // 推論結果をUIスレッド（メインスレッド）に反映させる。
                await MainActor.run {
                    self.ocrLines = results
                    self.isProcessing = false
                }
            } catch {
                // エラー発生時はメッセージをセットし、ユーザーに通知する。
                await MainActor.run {
                    self.errorMessage = "OCRに失敗しました: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }
}

// 画像の上に検出された行の矩形を描画するためのオーバーレイビュー。
struct BoundingBoxOverlay: View {
    // 表示対象の画像情報（アスペクト比計算に使用）。
    let image: NSImage
    // 描画する矩形データの配列。
    let ocrLines: [OCRLine]
    
    var body: some View {
        // 描画領域のサイズを取得するためのGeometryReader。
        GeometryReader { geo in
            // 元の画像サイズ。
            let imageSize = image.size
            // 現在のビューの表示サイズ。
            let viewSize = geo.size
            // 画像がどのように縮小表示されているかのスケールを計算。
            let scale = min(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
            // 画像が中央に配置される際、左右に生じる余白を計算。
            let offsetX = (viewSize.width - imageSize.width * scale) / 2
            // 画像が中央に配置される際、上下に生じる余白を計算。
            let offsetY = (viewSize.height - imageSize.height * scale) / 2
            
            // 各行のデータをループして矩形を描画。
            ZStack(alignment: .topLeading) {
                ForEach(ocrLines) { line in
                    BoxView(line: line, scale: scale, offsetX: offsetX, offsetY: offsetY)
                }
            }
        }
    }
}

// 検出された一つ一つの行に対して、枠線とプレビュー文字を表示するビュー。
struct BoxView: View {
    // 検出された行の情報。
    let line: OCRLine
    // 表示倍率。
    let scale: CGFloat
    // 横方向のオフセット。
    let offsetX: CGFloat
    // 縦方向のオフセット。
    let offsetY: CGFloat
    
    var body: some View {
        // 元の座標系での矩形。
        let rect = line.rect
        // 表示サイズに合わせてスケーリングされた幅。
        let w = rect.size.width * scale
        // 表示サイズに合わせてスケーリングされた高さ。
        let h = rect.size.height * scale
        // 表示サイズと中央寄せを考慮したX座標。
        let x = rect.origin.x * scale + offsetX
        // 表示サイズと中央寄せを考慮したY座標。
        let y = rect.origin.y * scale + offsetY
        
        ZStack(alignment: .topLeading) {
            // テキスト行を囲む枠線を描画。
            Rectangle()
                // ラベル（横書き/縦書き）に応じて色を変えて視認性を高める。
                .stroke(line.label == 1 ? Color.yellow : Color.green, lineWidth: 1.5)
                .frame(width: w, height: h)
            
            // 認識されたテキストがある場合、その冒頭一文字を枠の上に表示して内容を確認しやすくする。
            if !line.text.isEmpty {
                Text(String(line.text.prefix(1)))
                    .font(.system(size: max(8, 12 * scale)))
                    .foregroundColor(.white)
                    .padding(2)
                    .background(Color.black.opacity(0.6))
                    // 枠の少し上にずらして配置。
                    .offset(y: -15)
            }
        }
        // 指定した座標にビューの中心を配置する。
        .position(x: x + w/2, y: y + h/2)
    }
}
