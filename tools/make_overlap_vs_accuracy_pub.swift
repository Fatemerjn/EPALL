#!/usr/bin/env swift

import Foundation
import AppKit

struct Row {
    let method: String
    let x: Double
    let y: Double
}

struct QuadraticFit {
    let a: Double
    let b: Double
    let c: Double
    let xGrid: [Double]
    let yFit: [Double]
    let peakX: Double
    let peakY: Double
}

enum PlotError: Error, CustomStringConvertible {
    case missingColumn(String)
    case noData(String)
    case invalidRegression(String)
    case pdfContext(String)

    var description: String {
        switch self {
        case .missingColumn(let name):
            return "Input CSV is missing required column: \(name)"
        case .noData(let message):
            return message
        case .invalidRegression(let message):
            return message
        case .pdfContext(let message):
            return message
        }
    }
}

func parseDouble(_ value: String?) -> Double? {
    guard let value else { return nil }
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty || trimmed.lowercased() == "na" || trimmed.lowercased() == "none" {
        return nil
    }
    return Double(trimmed)
}

func parseCSVLine(_ line: String) -> [String] {
    var values: [String] = []
    var current = ""
    var inQuotes = false

    for char in line {
        if char == "\"" {
            inQuotes.toggle()
            continue
        }
        if char == "," && !inQuotes {
            values.append(current)
            current = ""
        } else {
            current.append(char)
        }
    }
    values.append(current)
    return values
}

func loadRows(inputPath: String) throws -> [Row] {
    let url = URL(fileURLWithPath: inputPath)
    let text = try String(contentsOf: url, encoding: .utf8)
    let lines = text.split(whereSeparator: \.isNewline).map(String.init)
    guard let headerLine = lines.first else {
        throw PlotError.noData("Input CSV is empty: \(inputPath)")
    }

    let headers = parseCSVLine(headerLine)
    guard let methodIndex = headers.firstIndex(of: "method") else {
        throw PlotError.missingColumn("method")
    }
    guard let xIndex = headers.firstIndex(of: "S_share_crit_ratio") else {
        throw PlotError.missingColumn("S_share_crit_ratio")
    }
    guard let yIndex = headers.firstIndex(of: "final_avg_acc") else {
        throw PlotError.missingColumn("final_avg_acc")
    }

    var rows: [Row] = []
    for line in lines.dropFirst() {
        let values = parseCSVLine(line)
        if values.count <= max(methodIndex, xIndex, yIndex) {
            continue
        }
        guard let x = parseDouble(values[xIndex]), let y = parseDouble(values[yIndex]) else {
            continue
        }
        let method = values[methodIndex].trimmingCharacters(in: .whitespacesAndNewlines)
        rows.append(Row(method: method.isEmpty ? "unknown" : method, x: x, y: y))
    }

    if rows.isEmpty {
        throw PlotError.noData("No valid rows with S_share_crit_ratio and final_avg_acc were found.")
    }
    return rows
}

func solve3x3(_ matrix: [[Double]], _ rhs: [Double]) -> (Double, Double, Double)? {
    var a = matrix
    var b = rhs
    let n = 3

    for i in 0..<n {
        var pivot = i
        var maxValue = abs(a[i][i])
        for r in (i + 1)..<n {
            let value = abs(a[r][i])
            if value > maxValue {
                maxValue = value
                pivot = r
            }
        }
        if maxValue < 1e-12 {
            return nil
        }
        if pivot != i {
            a.swapAt(i, pivot)
            b.swapAt(i, pivot)
        }

        let pivotValue = a[i][i]
        for c in i..<n {
            a[i][c] /= pivotValue
        }
        b[i] /= pivotValue

        for r in 0..<n where r != i {
            let factor = a[r][i]
            for c in i..<n {
                a[r][c] -= factor * a[i][c]
            }
            b[r] -= factor * b[i]
        }
    }

    return (b[0], b[1], b[2])
}

func quadraticFit(rows: [Row], points: Int = 240) throws -> QuadraticFit {
    guard rows.count >= 3 else {
        throw PlotError.invalidRegression("At least three points are required for a quadratic regression.")
    }

    let xs = rows.map(\.x)
    let ys = rows.map(\.y)
    let n = Double(rows.count)
    let sx = xs.reduce(0.0, +)
    let sx2 = xs.reduce(0.0) { partial, value in
        partial + value * value
    }
    let sx3 = xs.reduce(0.0) { partial, value in
        partial + value * value * value
    }
    let sx4 = xs.reduce(0.0) { partial, value in
        partial + value * value * value * value
    }
    let sy = ys.reduce(0.0, +)
    let sxy = zip(xs, ys).reduce(0.0) { partial, pair in
        partial + pair.0 * pair.1
    }
    let sx2y = zip(xs, ys).reduce(0.0) { partial, pair in
        partial + pair.0 * pair.0 * pair.1
    }

    let matrix = [
        [sx4, sx3, sx2],
        [sx3, sx2, sx],
        [sx2, sx, n],
    ]
    let rhs = [sx2y, sxy, sy]

    guard let solution = solve3x3(matrix, rhs) else {
        throw PlotError.invalidRegression("Quadratic regression failed because the system is singular.")
    }

    let a = solution.0
    let b = solution.1
    let c = solution.2
    let minX = xs.min() ?? 0.0
    let maxX = xs.max() ?? 1.0
    let xGrid = (0..<points).map { idx in
        minX + (maxX - minX) * Double(idx) / Double(max(points - 1, 1))
    }
    let yFit = xGrid.map { a * $0 * $0 + b * $0 + c }

    let actualBest = rows.max { lhs, rhs in lhs.y < rhs.y }!
    let peakX: Double
    if abs(a) > 1e-12 {
        let candidate = -b / (2.0 * a)
        peakX = min(max(candidate, minX), maxX)
    } else {
        peakX = actualBest.x
    }
    let peakY = a * peakX * peakX + b * peakX + c

    return QuadraticFit(a: a, b: b, c: c, xGrid: xGrid, yFit: yFit, peakX: peakX, peakY: peakY)
}

func drawText(
    _ text: String,
    at point: CGPoint,
    size: CGFloat,
    color: NSColor = .black,
    weight: NSFont.Weight = .regular,
    alignment: NSTextAlignment = .left,
    width: CGFloat = 400
) {
    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = alignment
    let attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: size, weight: weight),
        .foregroundColor: color,
        .paragraphStyle: paragraph,
    ]
    let string = NSAttributedString(string: text, attributes: attributes)
    let rect = CGRect(x: point.x, y: point.y, width: width, height: 36)
    string.draw(in: rect)
}

func drawMarker(context: CGContext, point: CGPoint, color: NSColor, style: Int, size: CGFloat) {
    context.saveGState()
    context.setLineWidth(1.0)
    context.setStrokeColor(NSColor.white.cgColor)
    context.setFillColor(color.cgColor)

    switch style % 4 {
    case 0:
        let rect = CGRect(x: point.x - size / 2, y: point.y - size / 2, width: size, height: size)
        context.fillEllipse(in: rect)
        context.strokeEllipse(in: rect)
    case 1:
        let rect = CGRect(x: point.x - size / 2, y: point.y - size / 2, width: size, height: size)
        context.fill(rect)
        context.stroke(rect)
    case 2:
        context.beginPath()
        context.move(to: CGPoint(x: point.x, y: point.y + size / 2))
        context.addLine(to: CGPoint(x: point.x - size / 2, y: point.y - size / 2))
        context.addLine(to: CGPoint(x: point.x + size / 2, y: point.y - size / 2))
        context.closePath()
        context.drawPath(using: .fillStroke)
    default:
        context.beginPath()
        context.move(to: CGPoint(x: point.x, y: point.y + size / 2))
        context.addLine(to: CGPoint(x: point.x - size / 2, y: point.y))
        context.addLine(to: CGPoint(x: point.x, y: point.y - size / 2))
        context.addLine(to: CGPoint(x: point.x + size / 2, y: point.y))
        context.closePath()
        context.drawPath(using: .fillStroke)
    }
    context.restoreGState()
}

func tickValues(min: Double, max: Double, count: Int) -> [Double] {
    guard count > 1 else { return [min] }
    if abs(max - min) < 1e-12 { return [min] }
    let step = (max - min) / Double(count - 1)
    return (0..<count).map { min + Double($0) * step }
}

func drawPlot(rows: [Row], outputPath: String) throws {
    let fit = try quadraticFit(rows: rows)
    let methods = Array(Set(rows.map(\.method))).sorted()
    let palette: [NSColor] = [
        NSColor(calibratedRed: 0.10, green: 0.31, blue: 0.62, alpha: 1.0),
        NSColor(calibratedRed: 0.76, green: 0.22, blue: 0.17, alpha: 1.0),
        NSColor(calibratedRed: 0.18, green: 0.55, blue: 0.34, alpha: 1.0),
        NSColor(calibratedRed: 0.55, green: 0.32, blue: 0.67, alpha: 1.0),
        NSColor(calibratedRed: 0.70, green: 0.45, blue: 0.11, alpha: 1.0),
    ]

    let page = CGRect(x: 0, y: 0, width: 740, height: 510)
    let plotRect = CGRect(x: 88, y: 82, width: 460, height: 348)
    let legendOrigin = CGPoint(x: 585, y: 358)

    let xs = rows.map(\.x)
    let ys = rows.map(\.y)
    let yAll = ys + fit.yFit
    var minX = xs.min() ?? 0.0
    var maxX = xs.max() ?? 1.0
    var minY = yAll.min() ?? 0.0
    var maxY = yAll.max() ?? 1.0
    let xPad = max(0.01, (maxX - minX) * 0.08)
    let yPad = max(0.005, (maxY - minY) * 0.14)
    minX = max(0.0, minX - xPad)
    maxX += xPad
    minY = max(0.0, minY - yPad)
    maxY += yPad

    func mapX(_ value: Double) -> CGFloat {
        let t = (value - minX) / max(maxX - minX, 1e-12)
        return plotRect.minX + CGFloat(t) * plotRect.width
    }
    func mapY(_ value: Double) -> CGFloat {
        let t = (value - minY) / max(maxY - minY, 1e-12)
        return plotRect.minY + CGFloat(t) * plotRect.height
    }

    let bestRegionHalfWidth = max(0.008, (maxX - minX) * 0.07)
    let regionMinX = max(minX, fit.peakX - bestRegionHalfWidth)
    let regionMaxX = min(maxX, fit.peakX + bestRegionHalfWidth)

    let url = URL(fileURLWithPath: outputPath)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    var mediaBox = page
    guard let context = CGContext(url as CFURL, mediaBox: &mediaBox, nil) else {
        throw PlotError.pdfContext("Failed to create PDF context for \(outputPath)")
    }

    context.beginPDFPage(nil)
    context.setFillColor(NSColor.white.cgColor)
    context.fill(page)

    let graphicsContext = NSGraphicsContext(cgContext: context, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = graphicsContext

    drawText(
        "Critical Shared Overlap vs Final Accuracy",
        at: CGPoint(x: plotRect.minX, y: 456),
        size: 16,
        weight: .semibold
    )
    drawText(
        "Quadratic fit highlights the non-linear peak in final performance.",
        at: CGPoint(x: plotRect.minX, y: 436),
        size: 11,
        color: NSColor(calibratedWhite: 0.25, alpha: 1.0)
    )

    context.saveGState()
    context.setFillColor(NSColor(calibratedRed: 0.95, green: 0.70, blue: 0.20, alpha: 0.16).cgColor)
    let highlightRect = CGRect(
        x: mapX(regionMinX),
        y: plotRect.minY,
        width: mapX(regionMaxX) - mapX(regionMinX),
        height: plotRect.height
    )
    context.fill(highlightRect)
    context.restoreGState()

    context.setStrokeColor(NSColor.black.cgColor)
    context.setLineWidth(1.0)
    context.move(to: CGPoint(x: plotRect.minX, y: plotRect.minY))
    context.addLine(to: CGPoint(x: plotRect.maxX, y: plotRect.minY))
    context.move(to: CGPoint(x: plotRect.minX, y: plotRect.minY))
    context.addLine(to: CGPoint(x: plotRect.minX, y: plotRect.maxY))
    context.strokePath()

    let xTicks = tickValues(min: minX, max: maxX, count: 5)
    for tick in xTicks {
        let x = mapX(tick)
        context.setStrokeColor(NSColor(calibratedWhite: 0.25, alpha: 1.0).cgColor)
        context.move(to: CGPoint(x: x, y: plotRect.minY))
        context.addLine(to: CGPoint(x: x, y: plotRect.minY - 5))
        context.strokePath()
        drawText(String(format: "%.3f", tick), at: CGPoint(x: x - 22, y: plotRect.minY - 24), size: 10)
    }

    let yTicks = tickValues(min: minY, max: maxY, count: 5)
    for tick in yTicks {
        let y = mapY(tick)
        context.setStrokeColor(NSColor(calibratedWhite: 0.25, alpha: 1.0).cgColor)
        context.move(to: CGPoint(x: plotRect.minX - 5, y: y))
        context.addLine(to: CGPoint(x: plotRect.minX, y: y))
        context.strokePath()
        drawText(String(format: "%.3f", tick), at: CGPoint(x: 18, y: y - 7), size: 10, width: 60)
    }

    drawText("Critical shared overlap ratio", at: CGPoint(x: plotRect.midX - 94, y: 34), size: 12)
    NSGraphicsContext.saveGraphicsState()
    context.translateBy(x: 22, y: plotRect.midY + 48)
    context.rotate(by: .pi / 2.0)
    drawText("Final average accuracy", at: CGPoint(x: 0, y: 0), size: 12)
    NSGraphicsContext.restoreGraphicsState()

    context.saveGState()
    context.setStrokeColor(NSColor(calibratedRed: 0.55, green: 0.10, blue: 0.10, alpha: 1.0).cgColor)
    context.setLineWidth(2.2)
    if let firstX = fit.xGrid.first, let firstY = fit.yFit.first {
        context.move(to: CGPoint(x: mapX(firstX), y: mapY(firstY)))
        for (x, y) in zip(fit.xGrid.dropFirst(), fit.yFit.dropFirst()) {
            context.addLine(to: CGPoint(x: mapX(x), y: mapY(y)))
        }
        context.strokePath()
    }
    context.restoreGState()

    for (methodIndex, method) in methods.enumerated() {
        let color = palette[methodIndex % palette.count]
        let subset = rows.filter { $0.method == method }
        for row in subset {
            drawMarker(
                context: context,
                point: CGPoint(x: mapX(row.x), y: mapY(row.y)),
                color: color,
                style: methodIndex,
                size: 10.5
            )
        }
    }

    let peakPoint = CGPoint(x: mapX(fit.peakX), y: mapY(fit.peakY))
    context.saveGState()
    context.setStrokeColor(NSColor(calibratedRed: 0.55, green: 0.10, blue: 0.10, alpha: 1.0).cgColor)
    context.setLineWidth(1.1)
    context.setLineDash(phase: 0, lengths: [5, 4])
    context.move(to: CGPoint(x: peakPoint.x, y: plotRect.minY))
    context.addLine(to: CGPoint(x: peakPoint.x, y: peakPoint.y))
    context.strokePath()
    context.restoreGState()

    drawMarker(
        context: context,
        point: peakPoint,
        color: NSColor(calibratedRed: 0.55, green: 0.10, blue: 0.10, alpha: 1.0),
        style: 0,
        size: 12
    )
    drawText(
        "Best-performing region",
        at: CGPoint(x: min(peakPoint.x + 18, plotRect.maxX - 120), y: min(peakPoint.y + 22, plotRect.maxY - 18)),
        size: 11,
        color: NSColor(calibratedRed: 0.42, green: 0.24, blue: 0.02, alpha: 1.0),
        weight: .medium,
        width: 140
    )
    drawText(
        String(format: "peak at %.3f", fit.peakX),
        at: CGPoint(x: min(peakPoint.x + 18, plotRect.maxX - 90), y: min(peakPoint.y + 8, plotRect.maxY - 34)),
        size: 10,
        color: NSColor(calibratedRed: 0.42, green: 0.24, blue: 0.02, alpha: 1.0),
        width: 100
    )

    drawText("Method", at: CGPoint(x: legendOrigin.x, y: legendOrigin.y + 34), size: 12, weight: .semibold)
    for (methodIndex, method) in methods.enumerated() {
        let y = legendOrigin.y - CGFloat(methodIndex * 22)
        drawMarker(
            context: context,
            point: CGPoint(x: legendOrigin.x + 8, y: y + 7),
            color: palette[methodIndex % palette.count],
            style: methodIndex,
            size: 10
        )
        drawText(method, at: CGPoint(x: legendOrigin.x + 22, y: y), size: 10, width: 130)
    }

    let legendYOffset = CGFloat(methods.count * 22 + 10)
    context.setStrokeColor(NSColor(calibratedRed: 0.55, green: 0.10, blue: 0.10, alpha: 1.0).cgColor)
    context.setLineWidth(2.2)
    context.move(to: CGPoint(x: legendOrigin.x, y: legendOrigin.y - legendYOffset))
    context.addLine(to: CGPoint(x: legendOrigin.x + 18, y: legendOrigin.y - legendYOffset))
    context.strokePath()
    drawText("Quadratic fit", at: CGPoint(x: legendOrigin.x + 22, y: legendOrigin.y - legendYOffset - 7), size: 10)

    let bandRect = CGRect(x: legendOrigin.x, y: legendOrigin.y - legendYOffset - 30, width: 18, height: 12)
    context.setFillColor(NSColor(calibratedRed: 0.95, green: 0.70, blue: 0.20, alpha: 0.16).cgColor)
    context.fill(bandRect)
    drawText("Peak region", at: CGPoint(x: legendOrigin.x + 22, y: legendOrigin.y - legendYOffset - 33), size: 10)

    NSGraphicsContext.restoreGraphicsState()
    context.endPDFPage()
    context.closePDF()
}

let inputPath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "results/thesis/overlap_vs_damage.csv"
let outputPath = CommandLine.arguments.count > 2
    ? CommandLine.arguments[2]
    : "results/thesis/report_plots/overlap_vs_accuracy_pub.pdf"

do {
    let rows = try loadRows(inputPath: inputPath)
    try drawPlot(rows: rows, outputPath: outputPath)
    FileHandle.standardOutput.write(Data("[INFO] Wrote publication plot: \(outputPath)\n".utf8))
    FileHandle.standardOutput.write(Data("[INFO] Rows plotted: \(rows.count)\n".utf8))
} catch {
    FileHandle.standardError.write(Data("[ERROR] \(error)\n".utf8))
    exit(1)
}
