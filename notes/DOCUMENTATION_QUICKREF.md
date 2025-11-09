# Documentation Quick Reference

## 🚀 Quick Start

### Generate Documentation
```bash
cd build
make doc
```

### View Documentation
```bash
# Linux
xdg-open docs/html/index.html

# macOS
open docs/html/index.html
```

### Generate and Open
```bash
make doc_open
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `Doxyfile` | Doxygen configuration |
| `CMakeLists.txt` | Build system with doc target |
| `docs/html/index.html` | Main documentation page |
| `DOCUMENTATION.md` | Complete documentation guide |
| `DOCUMENTATION_REVIEW.md` | What was documented |
| `DOCUMENTATION_SUMMARY.txt` | Statistics and overview |

## 📚 What's Documented

- ✅ **Tensor Class**: Core multi-dimensional array
- ✅ **Autograd**: Automatic differentiation
- ✅ **Linear Algebra**: Matrix/vector operations
- ✅ **Loss Functions**: MSE, CrossEntropy, BCE, L1
- ✅ **Optimizers**: SGD, Adam, RMSprop
- ✅ **Tensor Ops**: Broadcasting, reductions, softmax
- ✅ **GPU/BLAS**: Acceleration support
- ✅ **Statistics**: Mean, variance, correlation

## 🔍 Search Tips

In the documentation:
- Use the search box (top-right) for quick lookup
- Search by class name: `Tensor`, `Optimizer`
- Search by function: `matmul`, `backward`, `softmax`
- Search by namespace: `loss`, `linalg`, `tensor_ops`

## 📖 Common Lookups

| Topic | Search For |
|-------|------------|
| Creating tensors | `Tensor` constructor |
| Autograd usage | `backward`, `requires_grad` |
| Matrix operations | `linalg::matmul`, `linalg::inverse` |
| Loss functions | `loss::mse_loss`, `loss::cross_entropy_loss` |
| Training | `Optimizer`, `SGD`, `Adam` |
| Element-wise ops | `operator+`, `operator*`, `exp`, `log` |

## 💡 Documentation Features

- **Call Graphs**: See which functions call each other
- **Caller Graphs**: See where functions are used
- **Source Browser**: View source code inline
- **Cross-references**: Click to navigate related items
- **Code Examples**: Copy-paste ready examples

## 🔧 Customization

Edit `Doxyfile` to change:
- `PROJECT_NAME` - Project title
- `OUTPUT_DIRECTORY` - Output location
- `INPUT` - Which files to document
- `GENERATE_LATEX` - Enable PDF generation
- `HAVE_DOT` - Enable/disable graphs

## 📊 Statistics

- **37 HTML pages** generated
- **4.0 MB** documentation size
- **47 call graphs** with visual relationships
- **200+ functions** fully documented
- **20+ code examples** included

## 🛠️ Maintenance

### Update Documentation
1. Edit Doxygen comments in source
2. Run `make doc`
3. Check `docs/html/index.html`

### Clean Documentation
```bash
rm -rf docs/
```

### Rebuild from Scratch
```bash
cd build
rm -rf docs/
make doc
```

## 📝 Documentation Style

Use Doxygen comments:
```cpp
/**
 * @brief Brief description
 * @param name Parameter description
 * @return What it returns
 * @throws ExceptionType When thrown
 * @code
 * // Usage example
 * @endcode
 */
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `make doc` fails | Check Doxygen is installed: `which doxygen` |
| No graphs | Install Graphviz: `sudo apt install graphviz` |
| Missing pages | Check `INPUT` in Doxyfile includes your files |
| Broken links | Regenerate: `rm -rf docs && make doc` |

## 📧 Resources

- **Full Guide**: See `DOCUMENTATION.md`
- **What Changed**: See `DOCUMENTATION_REVIEW.md`
- **Statistics**: See `DOCUMENTATION_SUMMARY.txt`
- **Doxygen Manual**: https://www.doxygen.nl/manual/

---

**Generated**: 2024-11-08  
**Tool**: Doxygen 1.15.0  
**Format**: HTML with Graphviz graphs
