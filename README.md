# Hopfield Bipartition

Implementacja algorytmu Hopfielda do problemu dwupodziału grafu nieskierowanego z wagami. Model minimalizuje energię systemu, przypisując wierzchołki do jednej z dwóch grup.

## Zastosowanie

- Optymalizacja podziału grafów
- Przykład dynamicznego systemu energetycznego (sieć Hopfielda)
- Testowanie stabilności podziałów w grafach losowych i rzeczywistych

## Główne cechy

- Parametryzowalność: `c1`, `c2`, `max_iter`, `seed`, `check_interval`
- Powtarzalność wyników przez kontrolę generatora losowego
- Prosta wizualizacja podziału z wykorzystaniem NetworkX i Matplotlib
- Wbudowane testy jednostkowe sprawdzające poprawność działania i stabilność energii

## Uruchamianie

```bash
python hopfield.py

