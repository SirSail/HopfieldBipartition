# Hopfield Bipartition

Implementacja algorytmu Hopfielda do problemu dwupodziału grafu nieskierowanego z wagami. Model minimalizuje energię systemu, przypisując wierzchołki do jednej z dwóch grup.

## Zastosowanie

- Heurystyczna optymalizacja podziału grafów
- Przykład dynamicznego systemu energetycznego (sieć Hopfielda)
- Testowanie stabilności podziałów na losowych grafach

## Główne cechy

- Parametryzowalność:
  - `edge_weight_coefficient` — współczynnik wag krawędzi
  - `cut_penalty` — kara za przecięcie krawędzi
  - `max_iter` — maksymalna liczba iteracji optymalizacji
  - `seed` — deterministyczne wyniki przez generator losowy
  - `check_interval` — odstęp do zapisu energii (gdy `track_energy=True`)
- Prosta wizualizacja wyniku (kolorowe partycje)
- Wbudowane testy jednostkowe (pytest-style)
- Możliwość śledzenia historii energii (`get_energy_history()`)

## Uruchamianie

```bash
python hopfield.py

