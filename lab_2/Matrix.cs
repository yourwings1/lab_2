﻿using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace lab_2
{
    public class Matrix
    {
        public int Width { get; }
        public int Height { get; }

        private double[][] m;

        private static int vsize = Vector<double>.Count;

        public Matrix(int width, int height)
        {
            Width = width;
            Height = height;
            m = new double[height][];
            for (int row = 0; row < height; row++)
            {
                m[row] = new double[width];
            }
        }

        public Matrix Transpose()
        {
            var result = new Matrix(Height, Width);
            for (int row = 0; row < Width; row++)
            {
                for (int col = 0; col < Height; col++)
                {
                    result.m[col][row] = m[row][col];
                }
            }

            return result;
        }

        private Vector<double>[][] GetTransposedVectors()
        {
            Contract.Requires(Height % vsize == 0);

            var result = new Vector<double>[Width][];
            for (int row = 0; row < Width; row++)
            {
                var vectorrow = new Vector<double>[Height / vsize];
                result[row] = vectorrow;

                for (int col = 0; col < Height; col += vsize)
                {
                    var vectordata = new double[vsize];
                    for (int j = 0; j < vsize; j++) vectordata[j] = m[col + j][row];
                    vectorrow[col / vsize] = new Vector<double>(vectordata);
                }
            }

            return result;
        }

        public static Matrix Multiply(Matrix a, Matrix b)
        {
            Contract.Requires(a.Width == b.Height && a.Height == b.Width);
            Contract.Requires(a.Width % vsize == 0);

            var result = new Matrix(b.Width, a.Height);

            var bcolVectors = b.GetTransposedVectors();

            var arowVectors = new Vector<double>[result.Width / vsize];

            for (int row = 0; row < result.Height; row++)
            {
                var arow = a.m[row];
                var resultrow = result.m[row];

                for (int col = 0; col < result.Width; col += vsize) arowVectors[col / vsize] = new Vector<double>(arow, col);

                for (int col = 0; col < result.Width; col++)
                {
                    var bcol = bcolVectors[col];
                    double sum = 0;
                    for (int j = 0; j < result.Width / vsize; j++)
                        sum += Vector.Dot(arowVectors[j], bcol[j]);
                    resultrow[col] = sum;
                }
            }

            return result;
        }

        public static Matrix ParallelMultiply(Matrix a, Matrix b)
        {
            Contract.Requires(a.Width == b.Height && a.Height == b.Width);
            Contract.Requires(a.Width % vsize == 0);

            var result = new Matrix(b.Width, a.Height);

            var bcolVectors = b.GetTransposedVectors();

            Parallel.For(0, result.Height, (row) =>
            {
                var arow = a.m[row];
                var resultrow = result.m[row];

                var arowVectors = new Vector<double>[result.Width / vsize];
                for (int col = 0; col < result.Width; col += vsize) arowVectors[col / vsize] = new Vector<double>(arow, col);

                for (int col = 0; col < result.Width; col++)
                {
                    var bcol = bcolVectors[col];
                    double sum = 0;
                    for (int j = 0; j < result.Width / vsize; j++)
                        sum += Vector.Dot(arowVectors[j], bcol[j]);
                    resultrow[col] = sum;
                }
            });

            return result;
        }

        public void SetAt(int row, int col, double value)
        {
            m[row][col] = value;
        }

        public double GetAt(int row, int col)
        {
            return m[row][col];
        }

    }
}
