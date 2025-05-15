# Project CS6

## Overview
Project CS6 is an implementation for a comparative analysis between matrix multiplication algorithms: Standard O(n^3) and Strassen's divide-and-conquer algorithm. This project aims to become a benchmarking place to conduct benchmarks and tests between different matrix multiplication algorithms. It is designed to highlight key features and functionality for performance evaluation and algorithmic efficiency.

## Features
- Parameterized executables to cater to different benchmarks and tests.
- Graphical visualizations for performance comparison.
- Basic statistics for algorithmic efficiency analysis.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
  ```bash
  git clone https://github.com/chicocaine/project_cs6.git
  ```
2. Navigate to the project directory:
  ```bash
  cd project_cs6
  ```
3. Install dependencies:
  ```bash
  pip install:
    matplotlib numpy pandas tkinter
  ```

## Compilation
All code is compiled into one executable with:
```bash
    gcc -std=gnu99 -Wall -g3 -O3 -march=native -funroll-loops -ffast-math -fopenmp main.c standard.c strassen.c standard_block.c -o matrix_mul_benchmark_4.0.exe
```

## Usage
To run the project, execute the following sample command:
```bash
./matrix_mul_benchmark_4.0.exe -p 5 -w 10 -n -s -t 64 -b 16 -v -3 -o output.json
```
To customize the benchmark, add Parameters [options] after the execution:
* -p passes -w power
* -t threshold   use Strassen threshold t[input]>0
* -b blocksize   use blocked algorithm b[input]>0
* -n standard    runs standard algorithm count
* -s strassen    runs Strassen count
* -T threads     set thread count [default: 1]
* -1             disable timing
* -2             disable memory logging
* -3             check correctness
* -v             verbose output
* -V             visualize results
* -h             help\n"
* -o file[.json|.csv|.txt] output base/name and format

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
  ```bash
  git checkout -b feature-name
  ```
3. Commit your changes:
  ```bash
  git commit -m "Add feature-name"
  ```
4. Push to the branch:
  ```bash
  git push origin feature-name
  ```
5. Open a pull request.

## License
This project is licensed under the nothing. See no file for no details.

## Contact
For questions or feedback, please contact 'e.pechayco.546282@umindanao.edu.ph'.
