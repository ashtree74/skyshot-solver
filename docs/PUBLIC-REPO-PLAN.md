# Publiczny Skyshot Solver dla developerów

Decyzja zakresowa z 17 października 2026: `skyshot-solver` jest samodzielnym repo silnika i narzędzi developerskich. Obecny projekt aplikacyjny pozostaje bez zmian. Ten zakres zastępuje wcześniejszą koncepcję kopiowania całej aplikacji do tego publicznego repo.

## Aktualizacja wykonania — 17 października 2026

**Zakończony etap: publiczne Downloads.** Autor polecił publikację paczek na
`https://skyshot.app/downloads/`, z Full learned jako zalecanym wariantem i DEEP
opcjonalnie. Przygotowano nową wersję `2026.10.17.2`, ponownie kwalifikowaną na
90 rzeczywistych zdjęciach, z identycznymi statusami i WCS względem sprawdzonych
kopii. Pakiet `full-deep` nie wymusza pobrania wide. Doszedł downloader HTTPS
z wznawianiem Range, przypiętym katalogiem, SHA256 i instalacją do nowego katalogu.
Archiwa i wagi pozostają poza Git; w repo jest mały katalog wersji.
Wszystkie pięć archiwów pobrano przez publiczny HTTPS i zainstalowano od zera;
ponowiony odbiór 90 zdjęć zachował wyniki. Instrukcja pobrania i instalacji jest w
[DATA-INSTALL.md](DATA-INSTALL.md).
Poniższe zapisy o braku publikacji dotyczą wcześniejszych lokalnych kandydatów.

Natywny CI kodu z poprawką Windows przeszedł na macOS ARM64, Linux x86-64 i
Windows x86-64: [run 39056489660](https://github.com/ashtree74/skyshot-solver/actions/runs/39056489660).
Każdy job sprawdził 147 testów Rust, testy narzędzi Python, build release,
dokumentację, syntetyczny solve oraz pakowanie i uruchomienie CLI. Nie utożsamiamy
tego z odbiorem rzeczywistych zdjęć na każdej platformie; ten pozostaje macOS ARM64.


Doprecyzowanie autora: dane przygotowujemy do **użytku niekomercyjnego**, jawnie
wymieniając ich źródła, zależności, atrybucję i warunki. Nie traktujemy tej
inwentaryzacji jako ogólnej blokady prac ani dodatkowego procesu zatwierdzania.
Apache 2.0 pozostaje licencją kodu; warunki danych opisujemy oddzielnie.

Kontynuowany etap **B (dane)** oraz przygotowanie **C (odbiór CLI)**. Ekstrakcja
i matematyka solvera pozostały zachowane. Wszystkie 39 hashy plików pierwotnego
importu sprawdzono względem zapisanego commitu; hashe opisują oryginały przed
edycjami ekstrakcji. Nie synchronizujemy plików `AGENTS.md`, `agents.md`,
`CLAUDE.md`, `claud.md`, ich wariantów ani innych wskazówek agentowych.

Wykonane w tej kontynuacji:

- Przegląd pochodzenia D80 i LGQ-GD w [DATA-PROVENANCE.md](DATA-PROVENANCE.md),
  z oddzieleniem ustaleń od brakujących dokumentów i dowodów treningu. Wagi
  pozostają poza Git; lokalny eksport otrzymał nową tożsamość coder–indeks.
- Protokół oddzielnej dystrybucji w [DATA-RELEASE.md](DATA-RELEASE.md): niezmienne
  wersje, dokładna lista wersji silnika, manifesty plików, SHA256, pochodzenie,
  osobne paczki full/DEEP/wide, bez podmieniania mapowanych plików.
- Działające `tools/data_bundle.py create/verify/pack`. Lokalne pakowanie wymaga
  dokumentów źródeł i atrybucji; `pending` nie blokuje tej operacji i nie zmienia
  się automatycznie w potwierdzenie praw.
  `provenance/existing-full.bundle.json` zapisuje rzeczywiste hashe pełnego indeksu
  tylko jako audytowy draft z `pending`, bez danych i prywatnych ścieżek.
- Zapisano również hashe zawartości 1476 kafelków wejściowego D80 i rachunek
  lokalnego builda w `provenance/d80-local-inputs.json` oraz
  `provenance/local-lite-build.json`. To nie zastępuje brakujących warunków
  pierwotnego archiwum katalogu.
- Niezależnie zbudowany indeks handcrafted `lite` z D80 do nowego katalogu
  tymczasowego oraz solve lokalnego FITS M66, zgodny z wynikiem learned. Nie jest
  to publiczny fixture ani benchmark skuteczności.
- Lokalne pakowanie CLI z notices zależności i biblioteki standardowej Rust,
  hashami źródeł/binarki/archiwum, instrukcją instalacji i testem wypakowanej
  paczki. Uzupełniono brakujące licencje archiwum `rstar 0.12.2` z jego dokładnego
  commitu oraz teksty Rust 1.95.0 pominięte przez układ rustup. CI przypięto do
  sprawdzonego Rust 1.95.0, rozszerzono o pakowanie i test artefaktów; niczego nie
  opublikowano.
- Ponowny odbiór macOS i izolowanej kopii na Linux ARM64; szczegóły oraz granice
  w [raporcie weryfikacji](verification-2026-10-17.md). Windows i Linux x86-64
  nadal wymagają natywnego CI. Dodano CITATION i szablon zgłoszeń integratorów.

**Uzupełnienie etapu B:** przygotowano lokalny eksport LGQ-GD bez prywatnych
ścieżek, z autorem `Adam Jesionkiewicz` w metadanych i nowym hashem pary coder–JSON.
Wagi i tablice kwantyli pozostały identyczne. Nowe kopie full/DEEP/wide learned
oraz full handcrafted, osobny coder, manifesty, SHA256 i instalator są poza Git.
Instrukcja [DATA-INSTALL.md](DATA-INSTALL.md) prowadzi od wybranej paczki do
rozwiązania własnego obrazu. Niezależny build learned lite z wyeksportowanego
codera i solve prawdziwego M66 również przeszedł.

Porównanie obecnego silnika objęło 90 rzeczywistych zdjęć, 89 z WCS referencyjnym.
Oba pełne indeksy bez companionów: 86 poprawnych rozwiązań, zero błędnie
zaakceptowanych pól w tym zbiorze. Nowe kopie po oczyszczeniu metadanych dały
identyczne statusy i WCS. CLI z kompletem learned full/DEEP/wide: 89 poprawnych
rozwiązań, jedno `unsolved` bez referencji; 36 prób użyło wskazówek FITS, więc
nie utożsamiamy tego wyniku z testem blind. Wyniki i hashe prywatnego korpusu
pozostają lokalnie. Szerokie pola sprawdzono tylko na trzech obrazach 5,28–12,52°;
nie reklamujemy większego zakresu jako zweryfikowanego.

Sprawdzono SHA256 wszystkich pięciu archiwów i instalację z dołączonych skryptów
do świeżych katalogów we wszystkich czterech wariantach (`full`, `complete`,
`handcrafted`, `model`). Ponowiony odbiór wypakowanych plików dał identyczne
wyniki, a domyślna paczka full przeszła również bezpośredni solve FITS M66.
Sumy źródłowych indeksów pozostały zgodne, a ochrona `uchg` kanonicznego DEEP
pozostała włączona. Mały katalog wersji i sum jest w
`provenance/data-candidate-2026.10.17.1.json`; binaria są ignorowane przez Git.

**Pozostałe zadania techniczne:** publiczny zestaw rzeczywistych zdjęć do
kwalifikacji, odbiór danych full-sky na kolejnych platformach i test instalacji
przez zewnętrznych developerów. Hosting danych jest dostępny przez Downloads;
publiczny kod ma natywne CI na trzech systemach. Równolegle uzupełniamy jawny opis
pochodzenia i warunków danych. Wydanie binariów CLI, publikacja w rejestrach
oraz post na Cloudy Nights pozostają osobnymi poleceniami; `publish = false` trwa.

Następne integracje po podstawach: **trwałe żądania JSONL → małe C ABI → C++ i
.NET oraz rozszerzenie przykładu Python → ewentualny natywny binding Python**.
Kontrakt sesji i kryteria opisuje [RELEASE.md](RELEASE.md); żadnej z nowych
integracji nie przedstawiamy jako dostępnej.

Proponowany produkt: **kompletny, lokalny silnik astrometryczny, który można uruchomić z terminala albo osadzić we własnym programie**. CLI zapewnia szybki start, a biblioteka umożliwia wydajną integrację bez uruchamiania nowego procesu i mapowania indeksu dla każdej klatki. Otwieramy rzeczywistą implementację metody, nie atrapę wymagającą prywatnej usługi.

## Co trafia do repo

| Składnik | Po co go dajemy | Stan i priorytet |
|---|---|---|
| Rdzeń Rust | Ekstrakcja gwiazd, dominance/fingerprints, learned inference, indeks, wyszukiwanie, weryfikacja i dopasowanie WCS | Pierwsza kopia przygotowana |
| CLI | `solve`, `batch`, `extract`, `inspect`, `build`; JSON i WCS | Pierwsza wersja przygotowana |
| Biblioteka Rust | `Engine` utrzymujący indeks w pamięci; obrazy i gotowe listy gwiazd | Pierwsza wersja przygotowana; API przed stabilizacją |
| Matematyka WCS | Pixel↔sky, projekcje, SIP, rozwiązania szerokich pól | W kopii razem z testami |
| Małe syntetyczne demo | Pierwszy prawdziwy solve bez katalogu, zdjęć innych osób i konta | Przygotowane: generator indeksu i dopasowanej listy gwiazd |
| Testy i CI | Odtwarzalność, wykrywanie regresji, odbiór platform | Natywne CI macOS ARM64, Linux x86-64 i Windows x86-64 zaliczone; oddzielny odbiór danych full-sky na macOS |
| Dokumentacja integracji | Instalacja, dane, jednostki, wyniki, błędy, czas życia silnika | README, architektura, CLI i przykład Rust/Python przygotowane |
| Builder i opis formatu indeksu | Niezależność od infrastruktury autora i możliwość odtworzenia danych | Build handcrafted i learned lite oraz realne solve sprawdzone; wersjonowane Downloads z manifestami i opisem źródeł |
| Wagi modelu i model card | Korzystanie z tej samej metody, którą opisują artykuły | Publiczna paczka LGQ-GD i model card przez Downloads, poza Git; jawne braki pochodzenia |
| Trening i reprodukcja badań | Weryfikacja wyników i rozwój alternatywnych encoderów | Selektywny import potrzebnych modułów `lab`, bez całego historycznego laboratorium |
| C API i przykłady C++/.NET | Osadzenie w aplikacjach niezależnie od Rust | Następny etap po ustabilizowaniu podstawowego API |
| Python API | Badania, notebooki, automatyzacja | Teraz przykład subprocess; właściwy binding po kontrakcie sesji |
| Artykuły, autorstwo, cytowanie | Powiązanie metody z autorem i umożliwienie niezależnej oceny | Dwa istniejące PDF pozostają; CITATION i raport weryfikacji dodane |

Gotowe indeksy full/DEEP/wide są częścią oddzielnej dystrybucji danych przez Downloads,
z jawnym opisem źródeł i warunków dla użytku niekomercyjnego, i pozostają poza Git. W repo trafiają manifesty, SHA256,
zgodność wersji i instrukcja pobrania lub budowy. Kod buildera bez legalnej i
sprawdzonej ścieżki danych nie wystarcza do premiery dla użytkownika z własnymi zdjęciami.

## Co pozostaje poza tym repo

- Webowa aplikacja Skyshot, jej HTML/JS/CSS, interfejsy, anotacje, atlas prezentacyjny, strona produktowa i assety.
- Aplikacje macOS/iOS, menu bar, systemowy agent, zarządzanie folderami, aktualizator aplikacji i provisioning Apple.
- Produkcyjny serwer HTTP, konta, klucze usługowe, limity, płatności, email, panel administratora i analityka.
- Wystawca licencji, aktywacja, terminy ważności i rozdział Community/Pro.
- Eksperymentalna nawigacja, fuzja IMU i aplikacja mobilna: to odrębne produkty i kontrakty, nie warunek wydania solvera.
- Prywatne logi, stare wyniki, dane użytkowników, zdjęcia bez praw, infrastruktura i historia pierwotnego repo.

Kod obliczeniowy potrzebny do detekcji lub WCS pozostaje nawet wtedy, gdy aplikacja webowa też go używa. Granica przebiega według funkcji. Przykładem jest mały moduł transformacji współrzędnych wyodrębniony z warstwy współdzielonej, bez importowania atlasu i jego interfejsu.

## Architektura pierwszego wydania

Zaczynamy od jednego pakietu Cargo `skyshot-solver`, biblioteki `skyshot` i binarki `skyshot`. Nazwa pakietu nie jest jeszcze potwierdzona ani zarezerwowana w crates.io; `publish = false` pozostaje do właściwego wydania.

```text
src/                 rdzeń, Engine i cienkie CLI
examples/            wielokrotny solve, syntetyczny katalog, Python subprocess
tests/               testy izolowane, legalne małe fixtures
bench/               odtwarzalne benchmarki i generatory fixture
docs/                architektura, CLI, dane, integracja i roadmap
research/            istniejące artykuły z zachowaną licencją MIT
provenance/          commit i hashe źródłowego importu
.github/workflows/   niezależne CI
LICENSE              Apache 2.0 dla nowej implementacji
NOTICE               autorstwo i rozdzielenie warunków materiałów
```

Po przygotowaniu modelu dochodzą `models/` oraz ograniczony, odtwarzalny zestaw treningowy w `training/` lub `lab/`. Po przyjęciu kontraktu C dochodzą `bindings/c/` i przykłady. Nie tworzymy teraz kilku prawie pustych crate'ów ani gotowych obietnic dla wszystkich języków.

## Kontrakt dla integratorów

Najważniejsze operacje to: otwórz indeks, rozwiąż obraz, rozwiąż listę gwiazd, pobierz status i WCS, przelicz współrzędne, zamknij sesję. Indeks żyje dłużej niż pojedyncze zadanie. Biblioteka nie wymaga internetu, konta ani procesu agenta.

Trzeba jasno zdefiniować: jednostki RA/Dec i skali, początek układu pikseli, typy obrazów, znaczenie niepewności pozycji, status unsolved/ambiguous, opcjonalne projekcje, obsługę błędnego indeksu, liczenie budżetu czasu, wątki i anulowanie. Rust API pozostaje eksperymentalne do odbioru tych zasad. Format JSON ma własną wersję niezależną od szczegółów wewnętrznych statystyk.

Obecny `batch` przyjmuje kilka plików i używa jednej instancji silnika. Następny prosty punkt integracji to proces przyjmujący requesty JSONL przez stdin: jeden request → jeden wynik z ID, diagnostyka na stderr, ograniczenie rozmiaru wiadomości i czytelne zakończenie. Taki tryb może obsłużyć wiele języków przed powstaniem natywnych bindingów i nie wymaga publikowania serwera.

Małe C API powinno mieć opaque handle, jawne długości buforów, funkcję zwalniania wyników, wersję ABI, kody błędów oraz przechwytywanie panik na granicy FFI. Pierwsze przykłady: C++ oraz C# P/Invoke. Nie kopiujemy istniejącego mobilnego FFI razem z IMU, trackingiem kamery i aplikacją telefonu tylko po to, by nazwać je SDK solvera.

Python początkowo może korzystać z CLI, a przy pracy na wielu klatkach z batch. Binding powinien ostatecznie zachowywać sesję i przekazywać tablice pikseli/gwiazd bez plików pośrednich. Właściwy wybór opakowania nastąpi po sprawdzeniu kontraktu i odbiorców, bez utrzymywania równocześnie kilku alternatywnych bindingów.

## Dane i model jako część wiarygodnej premiery

Pierwsza kopia obsługuje zarówno ręczny klucz v1, jak i odczyt istniejącego indeksu learned z jego coderem. Domyślny nowy build bez wag tworzy wariant v1. Tego wyniku nie opisujemy liczbami z benchmarków LGQ-GD.

Przed premierą należy:

1. Przygotować wagi LGQ-GD z model card: pochodzenie, architektura, wejścia, trening, seed, parametry, wersja eksportu i hash. Usunąć prywatne ścieżki z opisów, zachowując zgodność model–indeks; nie zmieniać metadanych w istniejącym indeksie bez przeliczenia tożsamości i walidacji.
2. Wybrać potrzebny kod treningu i eksportu wraz z jego zależnościami. Opisać legalne pozyskanie wejść; nie kopiować wielkich shardów, checkpointów i prywatnych katalogów wyników.
3. Jawnie opisać źródła, atrybucję i warunki D80/Gaia oraz indeksów w niekomercyjnym zakresie. Nie utożsamiać licencji programu ASTAP z licencją katalogu.
4. Przygotować wersjonowane paczki indeksów, manifesty i sumy poza repo. Pierwsze uruchomienie ma prowadzić do konkretnego kompatybilnego zestawu, bez ręcznej pomocy autora.
5. Włączyć mały zestaw realnych zdjęć z potwierdzoną zgodą oraz znanymi rozwiązaniami referencyjnymi. Syntetyczne demo pozostaje osobnym testem procesu, nie substytutem realnego benchmarku.

Istniejące kanoniczne indeksy pozostają w głównym `work/` pierwotnego projektu. Odbiór może czytać je przez jawną ścieżkę; kopiowanie kodu nie przenosi danych, nie zdejmuje `uchg` i nie przebudowuje opublikowanego katalogu.

## Kolejność prac

| Etap | Rezultat | Orientacyjny wysiłek |
|---|---|---|
| A — początek wykonany | Wybrane źródła, CLI, Engine, testy, syntetyczne demo, dokumentacja i zakres | Pierwsza kopia w tym zadaniu |
| B — przygotowanie danych | Model, pochodzenie, receptura treningu, paczki indeksów i przykłady z atrybucją | 1–3 dni pracy; bez dodatkowej ogólnej blokady prawnej |
| C — odbiór developerski | Mac/Linux/Windows CI, binarki, notices zależności, benchmark wydania, instalacja przez 2–3 osoby | 1–2 dni po gotowości danych |
| D — pierwsza publikacja kodu i CLI | Tag, release, sumy, README z rzeczywistą tabelą platform i linkami | Po spełnieniu kryteriów poniżej |
| E — integracje | JSONL session, C API, C++/.NET, potem Python | Kolejne małe wydania według zgłoszeń integratorów |

Nie uzależniamy pierwszego kodu od C API, wheelów dla wszystkich platform, WASM, GUI ani kompletnego wsparcia N.I.N.A. Wrappery ASTAP i solve-field wymagają osobnych testów zgodności; same funkcje zapisu WCS nie oznaczają gotowej integracji.

## Kryteria przed publikacją implementacji

- [ ] Zatwierdzony diff importu i przegląd pochodzenia kodu; brak sekretów, prywatnych danych oraz kodu aplikacji webowej.
- [ ] Spójne Apache 2.0 dla nowego kodu, zachowane MIT wcześniejszych artykułów, autorstwo oraz komplet wymaganych notices w paczkach binarnych.
- [ ] Niezależny build i testy ze świeżego klona, bez pierwotnego projektu i prywatnego cache danych.
- [ ] Udokumentowany działający pierwszy solve syntetyczny oraz ścieżka realnego solve z kompatybilnym indeksem.
- [ ] Potwierdzony wynik CI i odbioru dla każdej reklamowanej platformy; niewykonanych testów nie oznaczamy jako zaliczone.
- [ ] Benchmark przypisany do kodu, modelu, danych i sprzętu: cold/warm, p50/p95, skuteczność, fałszywe akceptacje i limity badania.
- [ ] Stabilne artefakty z wersją i SHA256; podpisywanie lub platformowe instrukcje zgodne z rzeczywistym sposobem dystrybucji.
- [ ] CONTRIBUTING, CITATION, issue templates i kanał kontaktu dla integratorów; brak obietnic utrzymania wszystkich przyszłych bindingów.

## Rozwój po wyodrębnieniu

Po odbiorze publiczny solver powinien stać się źródłem wspólnego silnika, a aplikacje korzystać z jego wersji. Przestawienie obecnego produktu na tę zależność będzie osobnym zadaniem — teraz niczego w nim nie zmieniamy. Do tego momentu każdy kolejny import ma wskazany commit i przegląd różnic; nie kopiujemy całych katalogów na ślepo w obie strony.

Przekaz premiery: „możesz użyć Skyshot jako silnika we własnym narzędziu, sprawdzić metodę i zbudować całość lokalnie”. To daje społeczności konkretną wartość i wiąże projekt z autorem. Nowe aplikacje do stackowania, obróbki i gotowe interfejsy mogą rozwijać się osobno, bez blokad w opublikowanym solverze.
