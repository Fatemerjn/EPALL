#!/usr/bin/env swift

import Foundation
import AppKit

struct Row {
    let method: String
    let x: Double
    let y: Double
}

struct RegressionBand {
    let xGrid: [Double]
    let yFit: [Double]
    let yLower: [Double]
    let yUpper: [Double]
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

func tCritical95(df: Int) -> Double {
    let table: [Int: Double] = [
        1: 12.706204736432095,
        2: 4.302652729696142,
        3: 3.182446305284263,
        4: 2.7764451051977987,
        5: 2.570581835636305,
        6: 2.4469118511449692,
        7: 2.3646242510102993,
        8: 2.306004135204166,
        9: 2.2621571628540993,
        10: 2.2281388519649385,
        11: 2.200985160082949,
        12: 2.1788128296634177,
        13: 2.160368656461013,
        14: 2.1447866879169273,
        15: 2.131449545559323,
        16: 2.1199052992210112,
        17: 2.1098155778331806,
        18: 2.10092204024096,
        19: 2.093024054408263,
        20: 2.085963447265837,
        21: 2.079613844727662,
        22: 2.073873067904015,
        23: 2.068657610419041,
        24: 2.063898561628021,
        25: 2.0595385527532946,
        26: 2.0555294386428713,
        27: 2.0518305164802833,
        28: 2.048407141795244,
        29: 2.045229642132703,
        30: 2.042272456301238,
    ]
    return table[df] ?? 1.959963984540054
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
    guard let yIndex = headers.firstIndex(of: "avg_forgetting") else {
        throw PlotError.missingColumn("avg_forgetting")
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
        throw PlotError.noData("No valid rows with S_share_crit_ratio and avg_forgetting were found.")
    }
    return rows
}

func regressionBand(rows: [Row], points: Int = 200) throws -> RegressionBand {
    guard rows.count >= 2 else {
        throw PlotError.invalidRegression("At least two points are required for regression.")
    }

    let xs = rows.map(\.x)
    let ys = rows.map(\.y)
    let n = Double(rows.count)
    let xMean = xs.reduce(0, +) / n
    let yMean = ys.reduce(0, +) / n
    let sxx = zip(xs, xs).reduce(0.0) { partial, pair in
        let dx = pair.0 - xMean
        return partial + dx * dx
    }
    if abs(sxx) < 1e-12 {
        throw PlotError.invalidRegression("Cannot fit regression: S_share_crit_ratio has zero variance.")
    }

    let sxy = zip(xs, ys).reduce(0.0) { partial, pair in
        partial + (pair.0 - xMean) * (pair.1 - yMean)
    }
    let slope = sxy / sxx
    let intercept = yMean - slope * xMean
    let residuals = zip(xs, ys).map { x, y in y - (intercept + slope * x) }
    let dof = rows.count - 2
    let rss = residuals.reduce(0.0) { $0 + $1 * $1 }
    let residualStd: Double
    if dof > 0 {
        residualStd = sqrt(rss / Double(dof))
    } else {
        residualStd = sqrt(rss / max(1.0, n))
    }
    let tCrit = tCritical95(df: max(1, dof))

    let minX = xs.min() ?? 0.0
    let maxX = xs.max() ?? 1.0
    let xGrid = (0..<points).map { idx in
        minX + (maxX - minX) * Double(idx) / Double(max(points - 1, 1))
    }

    var yFit: [Double] = []
    var yLower: [Double] = []
    var yUpper: [Double] = []

    for x0 in xGrid {
        let fit = intercept + slope * x0
        let seMean = residualStd * sqrt((1.0 / n) + ((x0 - xMean) * (x0 - xMean) / sxx))
        let delta = tCrit * seMean
        yFit.append(fit)
        yLower.append(fit - delta)
        yUpper.append(fit + delta)
    }

    return RegressionBand(xGrid: xGrid, yFit: yFit, yLower: yLower, yUpper: yUpper)
}

func drawText(
    _ text: String,
    at point: CGPoint,
    size: CGFloat,
    color: NSColor = .black,
    weight: NSFont.Weight = .regular,
    alignment: NSTextAlignment = .left
) {
    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = alignment
    let attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: size, weight: weight),
        .foregroundColor: color,
        .paragraphStyle: paragraph,
    ]
    let string = NSAttributedString(string: text, attributes: attributes)
    let rect = CGRect(x: point.x, y: point.y, width: 400, height: 24)
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
    let band = try regressionBand(rows: rows)
    let methods = Array(Set(rows.map(\.method))).sorted()
    let palette: [NSColor] = [
        NSColor(calibratedRed: 0.10, green: 0.31, blue: 0.62, alpha: 1.0),
        NSColor(calibratedRed: 0.76, green: 0.22, blue: 0.17, alpha: 1.0),
        NSColor(calibratedRed: 0.18, green: 0.55, blue: 0.34, alpha: 1.0),
        NSColor(calibratedRed: 0.55, green: 0.32, blue: 0.67, alpha: 1.0),
        NSColor(calibratedRed: 0.70, green: 0.45, blue: 0.11, alpha: 1.0),
    ]

    let page = CGRect(x: 0, y: 0, width: 720, height: 500)
    let plotRect = CGRect(x: 88, y: 82, width: 450, height: 340)
    let legendOrigin = CGPoint(x: 570, y: 350)

    let xs = rows.map(\.x)
    let ys = rows.map(\.y)
    let yAll = ys + band.yLower + band.yUpper
    var minX = xs.min() ?? 0.0
    var maxX = xs.max() ?? 1.0
    var minY = yAll.min() ?? 0.0
    var maxY = yAll.max() ?? 1.0
    let xPad = max(0.01, (maxX - minX) * 0.06)
    let yPad = max(0.005, (maxY - minY) * 0.10)
    minX -= xPad
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
        "Critical Shared Overlap vs Average Forgetting",
        at: CGPoint(x: plotRect.minX, y: 450),
        size: 16,
        color: .black,
        weight: .semibold
    )

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
        drawText(String(format: "%.3f", tick), at: CGPoint(x: 22, y: y - 7), size: 10)
    }

    drawText("S_share_crit_ratio", at: CGPoint(x: plotRect.midX - 62, y: 34), size: 12)
    NSGraphicsContext.saveGraphicsState()
    context.translateBy(x: 22, y: plotRect.midY + 56)
    context.rotate(by: .pi / 2.0)
    drawText("avg_forgetting", at: CGPoint(x: 0, y: 0), size: 12)
    NSGraphicsContext.restoreGraphicsState()

    context.saveGState()
    context.setFillColor(NSColor.black.withAlphaComponent(0.10).cgColor)
    context.beginPath()
    if let firstX = band.xGrid.first, let firstUpper = band.yUpper.first {
        context.move(to: CGPoint(x: mapX(firstX), y: mapY(firstUpper)))
        for (x, upper) in zip(band.xGrid.dropFirst(), band.yUpper.dropFirst()) {
            context.addLine(to: CGPoint(x: mapX(x), y: mapY(upper)))
        }
        for (x, lower) in zip(band.xGrid.reversed(), band.yLower.reversed()) {
            context.addLine(to: CGPoint(x: mapX(x), y: mapY(lower)))
        }
        context.closePath()
        context.fillPath()
    }
    context.restoreGState()

    context.saveGState()
    context.setStrokeColor(NSColor.black.cgColor)
    context.setLineWidth(2.0)
    if let firstX = band.xGrid.first, let firstY = band.yFit.first {
        context.move(to: CGPoint(x: mapX(firstX), y: mapY(firstY)))
        for (x, fit) in zip(band.xGrid.dropFirst(), band.yFit.dropFirst()) {
            context.addLine(to: CGPoint(x: mapX(x), y: mapY(fit)))
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
                size: 10
            )
        }
    }

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
        drawText(method, at: CGPoint(x: legendOrigin.x + 22, y: y), size: 10)
    }
    let legendYOffset = CGFloat(methods.count * 22 + 10)
    context.setStrokeColor(NSColor.black.cgColor)
    context.setLineWidth(2.0)
    context.move(to: CGPoint(x: legendOrigin.x, y: legendOrigin.y - legendYOffset))
    context.addLine(to: CGPoint(x: legendOrigin.x + 18, y: legendOrigin.y - legendYOffset))
    context.strokePath()
    drawText("Linear fit", at: CGPoint(x: legendOrigin.x + 22, y: legendOrigin.y - legendYOffset - 7), size: 10)

    let bandRect = CGRect(x: legendOrigin.x, y: legendOrigin.y - legendYOffset - 28, width: 18, height: 12)
    context.setFillColor(NSColor.black.withAlphaComponent(0.10).cgColor)
    context.fill(bandRect)
    drawText("95% CI", at: CGPoint(x: legendOrigin.x + 22, y: legendOrigin.y - legendYOffset - 31), size: 10)

    NSGraphicsContext.restoreGraphicsState()
    context.endPDFPage()
    context.closePDF()
}

let inputPath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "results/thesis/overlap_vs_damage.csv"
let outputPath = CommandLine.arguments.count > 2
    ? CommandLine.arguments[2]
    : "results/thesis/report_plots/overlap_vs_forgetting_pub.pdf"

do {
    let rows = try loadRows(inputPath: inputPath)
    try drawPlot(rows: rows, outputPath: outputPath)
    FileHandle.standardOutput.write(Data("[INFO] Wrote publication plot: \(outputPath)\n".utf8))
    FileHandle.standardOutput.write(Data("[INFO] Rows plotted: \(rows.count)\n".utf8))
} catch {
    FileHandle.standardError.write(Data("[ERROR] \(error)\n".utf8))
    exit(1)
}
