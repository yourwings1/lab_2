using MKLNET;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Text;
using System.Numerics;
using lab_2;

class Program
{
    static void AssignRandom(Matrix matrix)
    {
        Random rnd = new Random();
        for (int row = 0; row < matrix.Height; row++)
        {
            for (int col = 0; col < matrix.Width; col++)
            {
                matrix.SetAt(row, col, (double)rnd.NextDouble());
            }
        }
    }

    // Метод для генерации случайной матрицы
    static double[,] GenerateRandomMatrix(int size)
    {
        Random random = new Random();
        double[,] matrix = new double[size, size];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                matrix[i, j] = random.NextDouble();
            }
        }
        return matrix;
    }

    // Метод для перемножения матриц с использованием функции cblas_dgemm из BLAS
    static double[,] MultiplyMatricesBLAS(double[,] matrix1, double[,] matrix2)
    {
        int size = matrix1.GetLength(0);
        double[,] result = new double[size, size];


        int m = size;
        int n = size;
        int k = size;
        double alpha = 1.0;
        double beta = 0.0;
        int lda = size;
        int ldb = size;
        int ldc = size;

        // Преобразование матриц в одномерные массивы
        double[] array1 = new double[size * size];
        double[] array2 = new double[size * size];
        double[] arrayResult = new double[size * size];

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                array1[i * size + j] = matrix1[i, j];
                array2[i * size + j] = matrix2[i, j];
            }
        }

        // Вызов функции cblas_dgemm
        Blas.gemm(Layout.RowMajor, Trans.No, Trans.No, m, n, k, alpha, array1, lda, array2, ldb, beta, arrayResult, ldc);
        // Преобразование одномерного массива обратно в матрицу
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                result[i, j] = arrayResult[i * size + j];
            }
        }

        return result;
    }

    static void Main(string[] args)
    {

        Console.OutputEncoding = Encoding.UTF8;
        int matrixSize = 4096;

        // Генерация случайных матриц
        double[,] matrix1 = GenerateRandomMatrix(matrixSize);
        double[,] matrix2 = GenerateRandomMatrix(matrixSize);

        Stopwatch stopwatch = new Stopwatch();

        Console.WriteLine("Кольцова Екатерина Владимировна\n090301-ПОВа-з21");

        var a = new Matrix(matrixSize, matrixSize);
        var b = new Matrix(matrixSize, matrixSize);

        AssignRandom(a);
        AssignRandom(b);

        //// Перемножение матриц с использованием стандартного алгоритма
        stopwatch.Start();
        var r1 = Matrix.Multiply(a, b);
        stopwatch.Stop();
        TimeSpan standardTime = stopwatch.Elapsed;

        Console.WriteLine("\nСтандартный алгоритм:");
        Console.WriteLine("Время: " + standardTime.TotalSeconds + " секунд");

        // Перемножение матриц с использованием функции cblas_dgemm из BLAS
        stopwatch.Reset();
        stopwatch.Start();
        double[,] resultBLAS = MultiplyMatricesBLAS(matrix1, matrix2);
        stopwatch.Stop();
        TimeSpan blasTime = stopwatch.Elapsed;

        Console.WriteLine("\nBLAS алгоритм:");
        Console.WriteLine("Время: " + blasTime.TotalSeconds + " секунд");

        // Перемножение матриц с использованием оптимизированного алгоритма
        stopwatch.Reset();
        stopwatch.Start();

        var r3 = Matrix.ParallelMultiply(a, b);
        stopwatch.Stop();
        TimeSpan optimizedTime = stopwatch.Elapsed;

        Console.WriteLine("\nОптимизированный алгоритм:");
        Console.WriteLine("Время: " + optimizedTime.TotalSeconds + " секунд");

        // Расчет сложности алгоритма
        double complexity = 2 * Math.Pow(matrixSize, 3);

        //// Расчет производительности в MFlops
        double standardMFlops = complexity / standardTime.TotalSeconds / 1e6;
        double blasMFlops = complexity / blasTime.TotalSeconds / 1e6;
        double optimizedMFlops = complexity / optimizedTime.TotalSeconds / 1e6;

        Console.WriteLine("Стандартный MFlops: " + standardMFlops);
        Console.WriteLine("BLAS MFlops: " + blasMFlops);
        Console.WriteLine("Оптимизированный MFlops: " + optimizedMFlops);
    }
}
