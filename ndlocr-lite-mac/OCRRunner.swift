import Foundation
import AppKit
import onnxruntime_objc
import Accelerate

// OCRの各行の結果を保持するための構造体。Identifiableに適合させることでSwiftUIのList等で扱いやすくしている。
struct OCRLine: Identifiable {
    // 各行を識別するための一意なID。
    let id = UUID()
    // 画像内でのテキスト行の位置とサイズを示す矩形。
    let rect: NSRect
    // 検出された矩形がテキストである確信度（スコア）。
    let score: Float
    // 検出されたオブジェクトの種類（1: 横書き、2: 縦書きなど）を示すラベル。
    let label: Int
    // 認識されたテキストの内容。初期値は空文字。
    var text: String = ""
}

// OCR（光学文字認識）の主要な処理を管理するクラス。
class OCRRunner {
    
    // アプリケーション起動時やデバッグ時に、テスト画像を用いてOCRプロセスを試走させるための静的メソッド。
    static func run() {
        // OCRが開始されたことをコンソールに通知。
        print("🚀 OCR開始")
        
        // メインスレッドをブロックしないよう、全ての重い処理をバックグラウンドのスレッドで行う。
        // 全ての重い処理をバックグラウンドで行う
        Task.detached(priority: .userInitiated) {
            // Assets.xcassetsに含まれる"test"という名前の画像を読み込む。
            guard let image = NSImage(named: "test") else {
                // 画像が見つからない場合はエラーを表示して処理を中断。
                print("❌ 画像 'test' が読み込めませんでした。Assets.xcassetsを確認してください。")
                return
            }

            do {
                // モデルの初期化（重い処理）を含めてインスタンスを生成。
                let runner = try OCRRunner()
                // 画像に対して検出から認識までの一連のプロセスを実行。
                let lines = try runner.process(image: image)
                
                // 最終的なOCR結果をコンソールに出力。
                print("--- OCR結果 ---")
                for line in lines {
//                    print("📍 [\(Int(line.rect.origin.x)), \(Int(line.rect.origin.y)), \(Int(line.rect.size.width)), \(Int(line.rect.size.height))] Score: \(String(format: "%.3f", line.score)) Label: \(line.label)")
                    // 認識されたテキストを一行ずつ表示。
                    print("📝 \(line.text)")
                }
                print("--------------")
                
            } catch {
                // プロセスの途中で発生したエラー（モデル不足、推論失敗など）を捕捉。
                print("❌ OCRエラー: \(error)")
            }
        }
    }
    
    // ONNX Runtimeの実行環境を管理するプロパティ。
    private let env: ORTEnv
    // 行検出用のモデル（DEIM）を保持するセッション。
    private let deimSession: ORTSession
    // 文字認識用モデル（PARSEQ）の、最大30文字まで対応するセッション。
    private let parseq30Session: ORTSession
    // 文字認識用モデル（PARSEQ）の、最大50文字まで対応するセッション。
    private let parseq50Session: ORTSession
    // 文字認識用モデル（PARSEQ）の、最大100文字まで対応するセッション。
    private let parseq100Session: ORTSession
    // モデルが出力するインデックスを実際の文字に変換するためのリスト。
    private let charList: [String]
    
    // OCRプロセスの初期設定。モデルの読み込みと文字リストの構築を行う。
    init() throws {
        // ONNX Runtimeの環境を警告レベルのログ設定で初期化。
        let env = try ORTEnv(loggingLevel: .warning)
        self.env = env
        
        // セッションの設定。CPUやGPUなどの実行デバイスを制御。
        let options = try ORTSessionOptions()
        // CoreMLでANEエラーが出るため、一旦無効化してCPUで動作確認
        // try options.appendExecutionProvider("CoreML")
        
        // 指定された名前のONNXモデルファイルを読み込み、セッションを生成するヘルパー関数。
        func createSession(name: String) throws -> ORTSession {
            // アプリケーションのバンドル内からモデルのパスを検索。
            guard let path = Bundle.main.path(forResource: name, ofType: "onnx") else {
                // モデルが見つからない場合は処理を中断。
                throw NSError(domain: "OCRRunner", code: 1, userInfo: [NSLocalizedDescriptionKey: "モデルが見つかりません: \(name)"])
            }
            // 指定されたパスのモデルを読み込み。
            return try ORTSession(env: env, modelPath: path, sessionOptions: options)
        }
        
        // 矩形検出用のモデルを読み込む。1024x1024ピクセルの入力サイズに最適化されている。
        self.deimSession = try createSession(name: "deim-s-1024x1024")
        // 文字列の長さに応じて使い分けるため、3種類の認識モデルを準備。
        self.parseq30Session = try createSession(name: "parseq-ndl-16x256-30-tiny-192epoch-tegaki3")
        self.parseq50Session = try createSession(name: "parseq-ndl-16x384-50-tiny-146epoch-tegaki2")
        self.parseq100Session = try createSession(name: "parseq-ndl-16x768-100-tiny-165epoch-tegaki2")
        
        // 文字リスト（対応文字集合）を定義したYAMLファイルを読み込む。
        // Load CharList
        guard let yamlPath = Bundle.main.path(forResource: "NDLmoji", ofType: "yaml"),
              let yamlString = try? String(contentsOfFile: yamlPath) else {
            // YAMLファイルが欠落している場合は認識ができないためエラー。
            throw NSError(domain: "OCRRunner", code: 2, userInfo: [NSLocalizedDescriptionKey: "NDLmoji.yamlが見つかりません"])
        }
        
        var list: [String] = []
        // YAML内の`charset_train`というキーに格納されている文字列を抽出するための正規表現。
        // Use regex to find charset_train and handle escaped quotes
        let pattern = #"charset_train:\s*\"((?:[^"\\]|\\.)*)\""#
        if let regex = try? NSRegularExpression(pattern: pattern, options: []),
           let match = regex.firstMatch(in: yamlString, options: [], range: NSRange(yamlString.startIndex..., in: yamlString)),
           let range = Range(match.range(at: 1), in: yamlString) {
            let charset = String(yamlString[range])
            // YAML特有のエスケープ文字（\" など）を元の文字に戻す処理。
            // Standard YAML escapes like \" -> "
            let unescaped = charset.replacingOccurrences(of: #"\""#, with: "\"")
                                   .replacingOccurrences(of: #"\\"#, with: #"\"#)
            // 文字列を一文字ずつの配列に変換。
            list = unescaped.map { String($0) }
        }
        // 読み込まれた文字リストを保持。
        self.charList = list
        // デバッグ用に読み込まれた文字数を出力。
        print("🧩 CharList Loaded: \(self.charList.count) chars")
    }
    
    // 入力画像に対して「検出」→「フィルタリング」→「認識」→「並べ替え」の一連のOCRパイプラインを実行する。
    func process(image: NSImage) throws -> [OCRLine] {
        // 1. 行検出（DEIMモデルを使用）：画像内のどこにテキストがあるかを探す。
        // 1. Line Detection (DEIM)
        var lines = try detectLines(image: image)
        
        // 2. フィルタリング：確信度が低いものや、テキスト以外の可能性が高いラベルを除去する。
        // 2. Filter: Only keep likely lines (Labels 1 and 2 in this model appear to be text)
        // Others (label 7, 9, 10 etc.) seem to be noise in many test cases
        lines = lines.filter { ($0.label == 1 || $0.label == 2) && $0.score > 0.3 }
        
        // 3. 非最大値抑制（NMS）：重複して検出された矩形の中から、最も確信度が高いものだけを残す。
        // 3. Remove significant overlaps (NMS-like)
        lines = nonMaximumSuppression(lines)
        
        // 4. 文字認識（PARSEQモデルを使用）：切り出された各行の画像から実際の文字列を読み取る。
        // 4. Text Recognition (PARSEQ)
        for i in 0..<lines.count {
            // 検出された矩形範囲を基に文字認識を実行。
            lines[i].text = try recognizeText(image: image, rect: lines[i].rect)
        }
        
        // 5. 読み取り順序でのソート：人間が読む順番（上から下、縦書きなら右から左）に並べ替える。
        // 5. Sort by Reading Order
        // Basically: Top-to-Bottom, then for vertical text Right-to-Left
        lines.sort { a, b in
            // ほぼ同じ高さ（Y座標の差が20px以内）にある場合、水平方向の順序で判断。
            // If they are on the same vertical level (roughly), sort by X
            if abs(a.rect.origin.y - b.rect.origin.y) < 20 {
                // 縦書き（ラベル2）の場合は、日本の伝統的な形式に従い右から左へと並べる。
                // If it's a vertical column area, right-to-left
                if a.label == 2 && b.label == 2 {
                    return a.rect.origin.x > b.rect.origin.x
                }
                // 横書きや混在の場合は左から右へ。
                return a.rect.origin.x < b.rect.origin.x
            }
            // 高さが異なる場合は、単純に上から下の順で並べる。
            return a.rect.origin.y < b.rect.origin.y
        }
        
        // 最終的に整理された認識結果の配列を返す。
        return lines
    }
    
    // 複数の矩形が重なっている場合に、確信度が低い方の矩形を削除して整理する。
    private func nonMaximumSuppression(_ lines: [OCRLine]) -> [OCRLine] {
        // まずスコアの高い順に並べ替え、確信度が高いものを優先的に処理する。
        var sorted = lines.sorted { $0.score > $1.score }
        var result: [OCRLine] = []
        
        // 候補がなくなるまで繰り返す。
        while !sorted.isEmpty {
            // 現在最もスコアが高いものを結果に追加。
            let best = sorted.removeFirst()
            result.append(best)
            // 追加した矩形と重なりが大きい他の候補を削除する。
            sorted.removeAll { other in
                // 矩形の重なり部分（交差領域）を計算。
                let intersection = best.rect.intersection(other.rect)
                // 重なりが全くなければ削除しない。
                if intersection.isEmpty { return false }
                // IoU (Intersection over Union): 重なり度合いを計算。
                let iou = (intersection.width * intersection.height) / (best.rect.width * best.rect.height + other.rect.width * other.rect.height - intersection.width * intersection.height)
                // 50%以上重なっている場合は、重複検出とみなして削除。
                return iou > 0.5 // Suppress if IoU > 0.5
            }
        }
        // 重複が取り除かれたリストを返す。
        return result
    }
    
    // MARK: - DEIM (Line Detection)
    
    // DEIMモデルを使用して画像からテキスト行の矩形を検出する。
    private func detectLines(image: NSImage) throws -> [OCRLine] {
        // モデルが期待する入力サイズ。
        let inputSize = CGSize(width: 800, height: 800)
        
        // 前処理：画像のアスペクト比を維持したまま、800x800の正方形に収めるためのスケーリング。
        // Preprocessing: Square padding & Resize
        let originalSize = image.size
        let maxDim = max(originalSize.width, originalSize.height)
        let scale = 800.0 / maxDim
        
        // NSImageからCGImageへの変換（ピクセル操作のため）。
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "OCRRunner", code: 3, userInfo: [NSLocalizedDescriptionKey: "CGImage変換失敗"])
        }
        
        // 余白（パディング）を含めた800x800の描画コンテキストを作成。
        // Create 800x800 padded image
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: nil, width: 800, height: 800, bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw NSError(domain: "OCRRunner", code: 4, userInfo: [NSLocalizedDescriptionKey: "Context作成失敗"])
        }
        
        // 中央に配置するためのオフセット計算。
        let drawWidth = originalSize.width * scale
        let drawHeight = originalSize.height * scale
        let xOffset = (800.0 - drawWidth) / 2.0
        let yOffset = (800.0 - drawHeight) / 2.0
        
        // 背景を黒で塗りつぶし、その上に画像をリサイズして描画。
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 800, height: 800))
        context.draw(cgImage, in: CGRect(x: xOffset, y: yOffset, width: drawWidth, height: drawHeight))
        
        // 加工された画像（パディング済みリサイズ画像）を取得。
        guard let paddedCGImage = context.makeImage() else {
            throw NSError(domain: "OCRRunner", code: 5, userInfo: [NSLocalizedDescriptionKey: "PaddedImage作成失敗"])
        }
        
        // 画像をモデルの入力形式（テンソル）に変換する。
        // Normalize (ImageNet) & CHW
        let pixelData = try getPixelData(cgImage: paddedCGImage)
        // 1枚の画像、3チャンネル(RGB)、800x800ピクセルの浮動小数点配列。
        var inputTensor = [Float](repeating: 0, count: 1 * 3 * 800 * 800)
        
        // 一般的な画像認識モデルで用いられるImageNet統計量に基づく標準化パラメータ。
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]
        
        // 各ピクセルの値を 0.0〜1.0 に正規化し、さらに平均と標準偏差で調整。
        for y in 0..<800 {
            for x in 0..<800 {
                let idx = (y * 800 + x) * 4
                for c in 0..<3 {
                    let val = Float(pixelData[idx + c]) / 255.0
                    // CHW形式（Channel, Height, Width）で格納。
                    inputTensor[c * 800 * 800 + y * 800 + x] = (val - mean[c]) / std[c]
                }
            }
        }
        
        // 推論の実行。
        // Inference
        // 画像データをテンソルオブジェクトにカプセル化。
        let imageTensor = try ORTValue(tensorData: NSMutableData(bytes: inputTensor, length: inputTensor.count * 4), elementType: .float, shape: [1, 3, 800, 800])
        // 元の画像サイズ情報を渡す（内部で矩形のスケール復元に使われることがある）。
        let sizeTensor = try ORTValue(tensorData: NSMutableData(bytes: [Int64(800), Int64(800)], length: 16), elementType: .int64, shape: [1, 2])
        
        // モデルを実行して、矩形、スコア、ラベルの出力を得る。
        let outputNames = try deimSession.outputNames()
        let outputs = try deimSession.run(withInputs: ["images": imageTensor, "orig_target_sizes": sizeTensor], outputNames: Set(outputNames), runOptions: nil)
        
        // 出力データを取り出す。
        let bboxesVal = outputs["boxes"]!
        let scoresVal = outputs["scores"]!
        let labelsVal = outputs["labels"]!
        
        // RawデータをData型として取得。
        let bboxes = try bboxesVal.tensorData() as Data
        let scores = try scoresVal.tensorData() as Data
        let labels = try labelsVal.tensorData() as Data
        
        // バイト列を適切な数値型（Float, Int64）の配列に変換。
        let bboxesFloat = bboxes.withUnsafeBytes { pointer in
            Array(pointer.bindMemory(to: Float.self))
        }
        let scoresFloat = scores.withUnsafeBytes { pointer in
            Array(pointer.bindMemory(to: Float.self))
        }
        let labelsInt64 = labels.withUnsafeBytes { pointer in
            Array(pointer.bindMemory(to: Int64.self))
        }
        
        var results: [OCRLine] = []
        let numBoxes = scoresFloat.count
        
        // モデルが見つけた各矩形を走査。
        for i in 0..<numBoxes {
            let score = scoresFloat[i]
            // 確信度が極端に低いものは無視する。
            if score < 0.2 { continue }
            
            // 矩形の座標。
            let x1 = bboxesFloat[i * 4]
            let y1 = bboxesFloat[i * 4 + 1]
            let x2 = bboxesFloat[i * 4 + 2]
            let y2 = bboxesFloat[i * 4 + 3]
            
            let label = Int(labelsInt64[i])
            
            // 800x800に加工される前の、元の画像座標系に変換するヘルパー。
            func revertX(_ x: Float) -> CGFloat {
                return CGFloat(x - Float(xOffset)) / CGFloat(scale)
            }
            func revertY(_ y: Float) -> CGFloat {
                return CGFloat(y - Float(yOffset)) / CGFloat(scale)
            }
            
            // 矩形オブジェクトを生成。
            var rect = NSRect(x: revertX(x1), y: revertY(y1), width: revertX(x2) - revertX(x1), height: revertY(y2) - revertY(y1))
            
            // 特定のケース（縦書きで非常に細い）における認識精度の向上のための調整。
            // 縦書きで極端に細い場合、少しだけ幅を広げてみる（文字の左右を削りすぎている可能性）
            if rect.size.height > rect.size.width && rect.size.width < 30 {
                let center = rect.origin.x + rect.size.width / 2.0
                let newWidth = max(rect.size.width, 32.0) // 32px程度あると認識しやすい
                rect.origin.x = center - newWidth / 2.0
                rect.size.width = newWidth
            }
            
            // 結果リストに追加。
            results.append(OCRLine(rect: rect, score: score, label: label, text: ""))
        }
        
        return results
    }
    
    // MARK: - PARSEQ (Text Recognition)
    
    // 行画像からテキストの内容を認識する。
    private func recognizeText(image: NSImage, rect: NSRect) throws -> String {
        // ピクセル操作のためにCGImageを取得。
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return "" }
        
        // 1. 画像の切り出し。
        // 指定された矩形範囲を行画像として抽出。
        // 1. Crop
        // ONNX coordinates are typically top-left. 
        // CGImage.cropping(to:) also typically uses top-left.
        let cropRect = CGRect(x: rect.origin.x, y: rect.origin.y, width: rect.size.width, height: rect.size.height)
        guard let cropped = cgImage.cropping(to: cropRect) else { return "" }
        
        // 2. 縦書き対応のための回転処理。
        // PARSEQモデルは横書き画像を前提としているため、縦書き行は左に90度回転させて横書きとして扱う。
        // 2. Rotation (Vertical Text)
        var targetImage = cropped
        let isVertical = rect.size.height > rect.size.width
        if isVertical {
            // Rotate 90 CCW
            targetImage = try rotate90CCW(cgImage: cropped)
        }
        
        // 3. 実行モデルの選択。
        // 文字列の長さに応じて、無駄の少ない最小のモデルを選択する。
        // 3. Select Model based on width/height ratio or char count estimation
        // For simplicity, let's use the 100 char model if very wide, otherwise smaller ones.
        // Actually, the guide says: "30字用: 256幅, 50字用: 384幅, 100字用: 768幅"
        // Let's decide based on width after resizing to height 16.
        let targetWidth = (16.0 / CGFloat(targetImage.height)) * CGFloat(targetImage.width)
        
        let session: ORTSession
        let modelWidth: Int
        if targetWidth <= 256 {
            // 短い行（約30文字以内）
            session = parseq30Session
            modelWidth = 256
        } else if targetWidth <= 384 {
            // 中程度の行（約50文字以内）
            session = parseq50Session
            modelWidth = 384
        } else {
            // 長い行（約100文字以内）
            session = parseq100Session
            modelWidth = 768
        }
        
        // 4. 画像のリサイズ。
        // モデルの入力仕様に合わせて、高さを16ピクセル、幅を選択したモデルの幅に調整。
        // 4. Resize to 16 x modelWidth
        guard let resized = try resizeCGImage(targetImage, width: modelWidth, height: 16) else { return "" }
        
        // 5. 正規化とテンソル変換。
        // ピクセル値を 0〜255 から -1.0〜1.0 の範囲に正規化。
        // 5. Normalization [-1, 1] & CHW
        let pixelData = try getPixelData(cgImage: resized)
        var inputTensor = [Float](repeating: 0, count: 1 * 3 * 16 * modelWidth)
        
        for y in 0..<16 {
            for x in 0..<modelWidth {
                let idx = (y * modelWidth + x) * 4
                for c in 0..<3 {
                    let val = Float(pixelData[idx + c])
                    // 127.5を中心とした正規化。
                    inputTensor[c * 16 * modelWidth + y * modelWidth + x] = (val / 127.5) - 1.0
                }
            }
        }
        
        // 推論の実行。
        // Inference
        let imageTensor = try ORTValue(tensorData: NSMutableData(bytes: inputTensor, length: inputTensor.count * 4), elementType: .float, shape: [1, 3, 16, modelWidth as NSNumber])
        
        let outputNames = try session.outputNames()
        let outputs = try session.run(withInputs: ["images": imageTensor], outputNames: Set(outputNames), runOptions: nil)
        
        // 推論結果（Logits: 各文字である確率の対数）の解析。
        // Output might be named "logits" or a numeric ID (e.g. "13469")
        // We take the first output since PARSEQ typically has one primary output tensor.
        guard let logitsVal = outputs.values.first else { return "" }
        
        let logits = try logitsVal.tensorData() as Data
        let logitsFloat = logits.withUnsafeBytes { pointer in
            Array(pointer.bindMemory(to: Float.self))
        }
        
        // 出力テンソルの形状（バッチ, 長さ, 文字セットサイズ）を確認。
        // Shape: [1, MaxLen, CharSetSize]
        let info = try logitsVal.tensorTypeAndShapeInfo()
        let shape = info.shape
        let maxLen = shape[1].intValue
        let charSetSize = shape[2].intValue
        
        var result = ""
        var maxIndices: [Int] = []
        // 各時間ステップ（文字位置）ごとに、最も確率が高い文字のインデックスを探す。
        for i in 0..<maxLen {
            let offset = i * charSetSize
            let slice = logitsFloat[offset..<(offset + charSetSize)]
            
            // 確率最大値を検索。
            if let maxIdx = slice.enumerated().max(by: { $0.element < $1.element })?.offset {
                maxIndices.append(maxIdx)
                // 0番目のインデックスは[EOS]（End of Sentence）を意味するため、そこで終了。
                if maxIdx == 0 { break } 
                
                // インデックスを実際の文字に変換（1番目以降が実際の文字リストに対応）。
                let charIdx = maxIdx - 1
                if charIdx >= 0 && charIdx < charList.count {
                    result += charList[charIdx]
                }
            }
        }
        
        // 認識結果が空で、何か推論されていた場合のデバッグ出力。
        if result.isEmpty && !maxIndices.isEmpty {
            print("🧩 Recognition Debug: Rect: \(rect), MaxIndices: \(maxIndices.prefix(10))...")
        }
        
        // 最終的な認識テキストを返す。
        return result
    }
    
    // MARK: - Helpers
    
    // CGImageから生のピクセルデータ（RGBA形式）を抽出する。
    private func getPixelData(cgImage: CGImage) throws -> [UInt8] {
        let width = cgImage.width
        let height = cgImage.height
        // ピクセルごとに4バイト（R, G, B, A）のメモリを確保。
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        // メモリ空間に直接描画するためのコンテキストを作成。
        let context = CGContext(data: &pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        // 指定されたピクセルメモリ領域に画像を描画。
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelData
    }
    
    // 画像を反時計回りに90度回転させる。縦書き文字を横向きにするために使用。
    private func rotate90CCW(cgImage: CGImage) throws -> CGImage {
        let width = cgImage.width
        let height = cgImage.height
        
        // 回転後は幅と高さが入れ替わる。
        // New dimensions: H x W
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: nil, width: height, height: width, bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return cgImage
        }
        
        // 座標系を変換して回転描画。
        context.translateBy(x: CGFloat(height), y: 0)
        context.rotate(by: .pi / 2.0)
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        
        // 回転後の画像を生成。
        return context.makeImage() ?? cgImage
    }
    
    // CGImageを任意のサイズにリサイズする。
    private func resizeCGImage(_ image: CGImage, width: Int, height: Int) throws -> CGImage? {
        // 新しいサイズのコンテキストを作成。
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0, space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(), bitmapInfo: image.bitmapInfo.rawValue)
        // 縮小時に画質が落ちないよう高品質な補完を設定。
        context?.interpolationQuality = .high
        // コンテキストの枠内に画像を拡大縮小して描画。
        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        // リサイズ後の画像を取得。
        return context?.makeImage()
    }
}

