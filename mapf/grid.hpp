// grid.hpp
//
// 4-connected grid world loaded from the MovingAI benchmark .map format
// (https://movingai.com/benchmarks/formats.html):
//
//   type octile
//   height <H>
//   width <W>
//   map
//   <H rows of W characters>
//
// Characters: '.' is passable terrain; '@' and 'T' (trees) are blocked.
// Any other character is rejected so a wrong or corrupted file fails
// loudly instead of silently producing a different world.
//
// Coordinates follow the MovingAI convention: x is the column, y is the
// row, (0,0) is the top-left corner. Agents move in 4-connected steps
// (up/down/left/right) or wait in place; both are enumerated by
// moves_from().
#pragma once

#include <cstddef>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mapf {

struct Cell {
    int x = 0;  // column
    int y = 0;  // row
    bool operator==(const Cell&) const = default;
};

class Grid {
public:
    static Grid parse(std::istream& in) {
        Grid grid;
        std::string token;
        int height = 0, width = 0;

        // Header: "type octile", "height H", "width W", "map".
        // height/width may appear in either order; "type" is informative.
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string key;
            ls >> key;
            if (key == "type") continue;
            if (key == "height") { ls >> height; continue; }
            if (key == "width") { ls >> width; continue; }
            if (key == "map") break;
            throw std::runtime_error("Grid::parse: unexpected header line: " + line);
        }
        if (height <= 0 || width <= 0) {
            throw std::runtime_error("Grid::parse: missing or invalid height/width");
        }

        grid.width_ = width;
        grid.height_ = height;
        grid.blocked_.assign(static_cast<std::size_t>(width) * height, false);
        for (int y = 0; y < height; ++y) {
            if (!std::getline(in, line) || static_cast<int>(line.size()) < width) {
                throw std::runtime_error("Grid::parse: map body shorter than declared size");
            }
            for (int x = 0; x < width; ++x) {
                char c = line[static_cast<std::size_t>(x)];
                if (c == '.') continue;
                if (c == '@' || c == 'T') {
                    grid.blocked_[grid.index(x, y)] = true;
                } else {
                    throw std::runtime_error(std::string("Grid::parse: unknown map char '") + c +
                                             "'");
                }
            }
        }
        return grid;
    }

    static Grid load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Grid::load: cannot open " + path);
        return parse(f);
    }

    int width() const { return width_; }
    int height() const { return height_; }

    bool in_bounds(int x, int y) const {
        return x >= 0 && x < width_ && y >= 0 && y < height_;
    }
    bool passable(int x, int y) const {
        return in_bounds(x, y) && !blocked_[index(x, y)];
    }
    bool passable(Cell c) const { return passable(c.x, c.y); }

    // The five legal moves from a cell: wait first, then the 4-connected
    // neighbors. Only passable destinations are returned.
    std::vector<Cell> moves_from(Cell c) const {
        std::vector<Cell> out;
        out.reserve(5);
        const int dx[] = {0, 1, -1, 0, 0};
        const int dy[] = {0, 0, 0, 1, -1};
        for (int k = 0; k < 5; ++k) {
            Cell to{c.x + dx[k], c.y + dy[k]};
            if (passable(to)) out.push_back(to);
        }
        return out;
    }

private:
    std::size_t index(int x, int y) const {
        return static_cast<std::size_t>(y) * width_ + x;
    }

    int width_ = 0;
    int height_ = 0;
    std::vector<bool> blocked_;
};

}  // namespace mapf
