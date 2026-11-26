# custom-ai

Serveur d'inférence local pour **Qwen3-4B-Instruct-2507** sur GPU NVIDIA. Le moteur local **Candle** et la gateway HTTP sont écrits en **Rust**. Le chemin **vLLM** reste disponible pour préparer le déploiement Kubernetes.

```text
Client → gateway Rust :8080 → moteur :8000 → Quadro RTX 5000
                             Candle ou vLLM
```

## Démarrage local en Rust

Prérequis : Linux, Rust, pilote NVIDIA fonctionnel (`nvidia-smi`) et CUDA Toolkit 12.8 (`nvcc --version`). Les poids du modèle restent dans leur dossier existant. Aucun environnement Python, Docker ou Kubernetes n'est nécessaire pour ce mode.

```bash
CUDA_COMPUTE_CAP=75 cargo build --release -p inference --features cuda --locked
./target/release/inference
```

La première compilation prépare les noyaux pour Turing (`sm_75`) et peut prendre plusieurs minutes. Le serveur charge les trois fichiers safetensors en float16 et conserve le modèle sur le GPU. La gateway écoute `127.0.0.1:8080` et le moteur interne `127.0.0.1:8000`.

Pour choisir un autre emplacement ou une autre adresse :

```bash
./target/release/inference --model /chemin/vers/Qwen3-4B-Instruct-2507 --bind 127.0.0.1:8080
```

Ce moteur local expose `/v1/models` et `/v1/chat/completions`, avec réponses JSON ou SSE. Il accepte les messages texte `system`, `developer`, `user`, `assistant` et `tool`, `max_tokens` (ou `max_completion_tokens`), `temperature` et `stream`. Les appels d'outils de type `function` utilisent le template fourni avec Qwen : définitions `tools`, historique `tool_calls`, résultats associés par `tool_call_id`, choix `auto`, `none`, `required` ou fonction nommée et option `parallel_tool_calls`. Les balises générées sont converties en appels OpenAI JSON ou SSE ; un appel incomplet, invalide ou inconnu est signalé comme une erreur. `stream_options.include_usage` ajoute le bilan des tokens en fin de flux. `reasoning_effort` est accepté pour la compatibilité avec OpenCode, sans effet sur ce modèle Instruct qui ne possède pas de mode de raisonnement réglable. Les images, schémas d'outils stricts et contraintes JSON ne sont pas pris en charge ; les autres champs inconnus sont refusés.

Une seule génération utilise le GPU à la fois, avec huit requêtes en attente au maximum. Le contexte par défaut est de **28 672 tokens partagés entre le prompt et la réponse**. Sans `max_tokens`, la réponse peut utiliser tout le contexte restant et s'arrête normalement au token de fin du modèle. Un préfixe exact de 2 048 tokens par défaut peut être réutilisé entre requêtes ; les préfixes différents ne sont jamais mélangés. Pour les très longues entrées, le snapshot est libéré afin de conserver de la marge VRAM. Le préremplissage utilise des blocs de 128 tokens et des points de synchronisation bornent le travail GPU en attente. Ctrl+C ou SIGTERM arrête le service.

## Discuter avec le modèle

Une fois le serveur lancé, ouvrez un second terminal dans le dépôt :

```bash
cargo run -p cli -- chat
```

Tapez directement votre question après `Vous >` puis Entrée. La réponse s'affiche progressivement et les échanges précédents sont conservés pour la prochaine question. `/reset` efface la conversation et `/quit` termine le client. Pour un seul prompt :

```bash
cargo run -p cli -- chat "Explique-moi simplement ce qu'est un réseau de neurones."
```

Pour un document long ou multiligne : `cargo run -p cli -- chat --file prompt.txt`. La limite de sortie est automatique par défaut. Options utiles : `--max-tokens 256`, `--temperature 0` pour une réponse déterministe et `--url http://127.0.0.1:8080`. Le client reprend `GATEWAY_API_KEY` si la gateway exige une clé.

## Choisir Qwen local dans OpenCode

Le fichier [opencode.json](opencode.json) déclare **Qwen3 4B local**, avec l'identifiant `customai/qwen3-4b-instruct`, comme modèle par défaut du projet. Il utilise directement l'API Rust sur `http://127.0.0.1:8080/v1`. OpenCode fournit l'interface et exécute les outils ; le modèle reste chargé sur le GPU par CustomAI.

Si le serveur n'est pas déjà lancé, démarrez-le dans un premier terminal :

```bash
./target/release/inference --max-model-len 28672 --prefix-cache-tokens 7680
```

Dans un second terminal :

```bash
cd /home/rustspeak/custom-ai
opencode
```

Tapez `/models`, puis choisissez **Qwen3 4B local**. Pour un essai en ligne de commande :

```bash
opencode run --model customai/qwen3-4b-instruct "Bonjour !"
```

La configuration annonce 28 672 tokens de contexte et 2 048 tokens de sortie par appel, conformément au profil utilisé pour cette intégration. Si vous changez la fenêtre du moteur, ajustez aussi `limit.context` dans la configuration. Les définitions d'outils et l'historique occupent une partie du contexte. Les options de raisonnement sont désactivées pour ce modèle Instruct. Sans `GATEWAY_API_KEY`, aucune clé n'est nécessaire ; si la gateway est protégée, ajoutez `"apiKey": "{env:GATEWAY_API_KEY}"` dans les `settings` du fournisseur.

Pour afficher le fournisseur dans les autres projets, ajoutez le bloc `providers.customai` à `~/.config/opencode/opencode.json` en conservant vos autres réglages. Cette inscription globale a été effectuée sur la machine de validation. Relancez OpenCode après une modification de sa configuration.

Le moteur local n'impose ni crédits, ni quota journalier, ni plafond de nombre total d'appels. Cette absence de quota ne supprime pas les limites techniques : contexte partagé de 28 672 tokens, sortie de 2 048 tokens par appel dans cette configuration OpenCode, une génération GPU active et huit requêtes en attente au maximum. Les outils qui utilisent un service externe conservent les conditions et les quotas de ce service.

La validation avec OpenCode **v2.2.6** et le GPU local couvre une réponse en streaming, puis une lecture de fichier par `read`, une création par `write` et une réponse finale reprenant le code lu. Les fichiers d'essai sont placés sous `.local/`, ignoré par Git. La prise en charge du protocole ne garantit pas la réussite de toutes les tâches de programmation avec un modèle de 4 milliards de paramètres.

L'[audit natif détaillé](benchmarks/opencode-local-validation.json) vérifie également `glob`, `grep`, `edit`, `shell`, `execute` et `webfetch`, ainsi que le chargement des **16 compétences annoncées** suivi d'une confirmation par Qwen. Les résultats des outils et les fichiers créés sont contrôlés indépendamment du texte final du modèle. `websearch` atteint l'outil mais demande le choix d'une intégration de recherche, que le client non interactif ne peut pas fournir. Les questions interactives, la délégation et la génération d'images sont couvertes au niveau du transport générique des appels, sans exécution réelle dans cet audit. Charger une compétence ne vérifie pas toutes ses tâches : ses intégrations externes restent nécessaires, et ce modèle ne comprend pas les images.

Le réglage initial `--max-model-len 15360` désactivait le cache au-delà de 7 680 tokens, alors que le prompt OpenCode mesuré en contenait déjà 7 768. Avec le profil ci-dessus, 7 680 tokens de préfixe commun sont réutilisés. Sur trois requêtes identiques avant et trois après, la durée moyenne après prise en charge GPU passe de **10,36 s à 0,76 s** pour les deux requêtes chaudes après modification, avec le même texte de réponse. Le premier appel froid reste à **10,51 s**. Ces durées soustraient la latence jusqu'aux en-têtes pour distinguer le service GPU de l'attente en file ; les valeurs complètes sont dans [la mesure OpenCode](benchmarks/opencode-local-performance.json). Le cache accélère la préparation répétée, pas le débit de décodage d'une longue réponse. Il conserve environ 1 080 Mio de KV pour ce préfixe ; les prompts dépassant la moitié du contexte continuent de libérer ce snapshot pour protéger la VRAM.

## Agent de recherche et navigation en Rust

La CLI possède une boucle d'agent qui utilise le modèle local et pilote **Chrome ou Firefox** avec le protocole [WebDriver](https://www.w3.org/TR/webdriver2/). Le client, la lecture du DOM, le décodage des liens et l'orchestration sont écrits en Rust : aucun Python, Node.js, serveur MCP ou script JavaScript ajouté n'est nécessaire. Cette CLI conserve son protocole d'actions JSON, distinct des appels natifs `tools` et `tool_calls` utilisés par OpenCode.

Prérequis : Chrome avec un [ChromeDriver compatible](https://developer.chrome.com/docs/chromedriver/get-started), ou Firefox avec [GeckoDriver](https://firefox-source-docs.mozilla.org/testing/geckodriver/Usage.html). Les pilotes sont recherchés dans `.local/bin/`, puis dans `PATH`, et peuvent être indiqués avec `--driver /chemin/du/pilote`. Sur la machine de validation : Chrome et ChromeDriver `157.0.8103.38`, Firefox `159.0` et GeckoDriver `0.37.1`. Les pilotes téléchargés dans `.local/bin/` sont ignorés par Git ; un clone du dépôt nécessite leur installation. Les navigateurs utilisent un profil temporaire séparé du profil personnel.

```bash
cargo build -p cli --locked

# Recherche avec moteurs de secours, sans appeler le modèle
./target/debug/cli search "langage Rust documentation officielle" --browser chrome
./target/debug/cli search "langage Rust documentation officielle" --browser firefox

# L'agent choisit ses recherches et les pages à lire, puis cite les sources
./target/debug/cli agent "Recherche et compare les avantages de Rust et Go, avec des sources." --browser chrome
./target/debug/cli agent "Recherche et compare les avantages de Rust et Go, avec des sources." --browser firefox

# Lecture directe d'une page publique, sans modèle
./target/debug/cli browse https://doc.rust-lang.org/book/ --browser firefox

# Une URL fournie reste utilisable si les moteurs sont indisponibles
./target/debug/cli agent "Résume la source fournie et cite-la." \
  --source-url https://doc.rust-lang.org/book/ \
  --browser chrome --report /tmp/custom-ai-recherche.json
```

Chaque action `search` essaie **Google et Bing**, puis **DuckDuckGo et Brave** si aucun résultat utile n'est obtenu. Si ces moteurs restent indisponibles, les API publiques de **Wikipédia en français et en anglais** servent de secours encyclopédique. Ce dernier n'est pas un index général du Web : il ne convient pas à toutes les questions, notamment aux prix et aux événements très récents. Les CAPTCHA détectés sont mémorisés pendant la session afin d'éviter de refaire la même tentative bloquée à chaque requête. L'agent peut reformuler ses recherches, changer de langue ou utiliser `site:...` pour cibler une source.

Pour une recherche généraliste par API, la variable `BRAVE_SEARCH_API_KEY` active en priorité [Brave Search API](https://api-dashboard.search.brave.com/documentation/guides/authentication). La clé reste dans l'environnement ; elle n'est ni sauvegardée dans un `.env` ni incluse dans les rapports. Si cette API échoue, les autres moteurs restent utilisés. Sans clé, les moteurs publics et Wikipédia restent accessibles selon leurs restrictions. Le chemin Brave API est testé avec des données simulées ; aucune clé n'était disponible pour un essai réseau authentifié. Pour saisir une clé dans zsh :

```bash
read -rs 'BRAVE_SEARCH_API_KEY?Clé Brave Search : '
export BRAVE_SEARCH_API_KEY
```

Les résultats et les liens découverts reçoivent des identifiants stables, dans une limite de 32 sources par travail. L'agent dispose de `search`, `open` pour une URL HTTP(S) publique, `read` sur un identifiant et `answer`. La lecture d'une page découvre ses liens, qui peuvent ensuite être suivis. Une URL proposée par le modèle est marquée comme telle et devient une source citable seulement après lecture réussie. La réponse finale doit citer des pages réellement lues ; la CLI ajoute elle-même leurs URL. Cela vérifie la provenance des références, pas la véracité de chaque affirmation. Les actions refusées, exécutées et échouées sont distinguées dans les rapports.

La lecture se fait par passages de 4 000 octets UTF-8 avec `offset`, `next_offset`, `total_bytes` et un indicateur de troncature. L'option d'outil `find` cherche une expression exacte dans le texte disponible pour lire un passage plus loin dans la page. Le texte extrait est limité à 160 000 octets par page ; les extraits conservés dans le contexte du modèle partagent un budget ajusté à sa fenêtre. Les articles Wikipédia sont d'abord lus via leur API publique de texte, avec repli sur le navigateur si nécessaire ; le champ `reader` distingue `mediawiki-api` de `webdriver`. Ce texte est un extrait fourni par Wikipédia, pas une copie de tous les éléments visuels de l'article. Un CAPTCHA ou une page de vérification reconnus ne sont pas considérés comme une source lue. Les pages rendues sous forme de texte sont prises en charge ; les images, vidéos, connexions privées et paywalls ne sont pas rendus accessibles par ces outils.

Le travail est borné par défaut à **16 décisions, 4 recherches, 8 lectures de passages et 600 secondes**. Le délai d'appel au modèle suit `--timeout`, et le travail complet reste borné par ce même délai : une limite interne fixe de 120 secondes ne coupe plus prématurément les générations lentes. Une question d'agent accepte 4 000 octets au maximum. `--file question.txt` conserve les guillemets et les retours à la ligne. Options : `--max-steps`, `--max-searches`, `--max-reads`, `--max-tokens` (1 024 par décision par défaut), `--timeout`, `--navigation-timeout`, `--browser-binary`, `--report`. Les rapports intermédiaires conservent l'état `in_progress` ; une fin réussie est marquée `complete`, un échec détecté `failed`. Le mode `chat` conserve son comportement et son contexte habituel. Pour approfondir :

```bash
./target/debug/cli agent "Recherche mon sujet, recoupe les sources et indique les limites." \
  --browser firefox --max-steps 32 --max-searches 8 --max-reads 16 --timeout 1800

# Lire un passage plus loin dans une page connue
./target/debug/cli browse https://doc.rust-lang.org/book/ --find "Installation" --browser chrome
```

**Essais du 24 novembre 2026 :** Google et DuckDuckGo demandent un CAPTCHA, Brave affiche également une vérification et Bing ne fournit pas de résultats utilisables sur la connexion de test. Le basculement public aboutit néanmoins à **neuf résultats encyclopédiques** pour « langage Rust » : cinq en français et quatre en anglais. Cela valide un secours fonctionnel, pas la disponibilité de toute la recherche Web. Pour intervenir manuellement dans une fenêtre de navigateur :

```bash
./target/debug/cli agent "Recherche la documentation officielle de Rust." --browser chrome --visible
```

En mode visible, la recherche attend jusqu'à 90 secondes pour permettre de traiter un CAPTCHA ou un consentement ; elle ne les contourne pas. Une session graphique est nécessaire. `--source-url` permet de poursuivre sur une source connue malgré ces limites. Les erreurs de navigation, de modèle, de protocole et les dépassements de durée ferment la session ; Ctrl+C annule le travail. Les URL sont limitées à HTTP(S) public et contrôlées avant la navigation et après la redirection. Ce contrôle ne constitue pas un filtrage de toutes les sous-requêtes réseau d'une page.

Les tests Rust utilisent notamment des serveurs HTTP et des outils simulés pour vérifier la boucle de décisions, les moteurs de secours, l'exploration des liens, les passages, les refus d'actions inconnues, les citations, les limites, les erreurs et la fermeture WebDriver. La lecture réelle a réussi dans Chrome et Firefox ; voir [la validation initiale](benchmarks/web-tools-validation.json). Sur la machine de développement, le linker LLVM a planté à l'édition des liens ; la compilation et les tests passent avec le linker GNU : `RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo build -p cli --locked` et la même variable pour `cargo test --workspace --locked`.

La [validation élargie en conditions réelles](benchmarks/web-research-resilience.json) vérifie aussi une recherche autonome sans URL fournie : recherche, découverte d'un article, lecture par API et réponse citée, en trois décisions. Un second test sur une URL explicitement fournie vérifie l'exploitation du contenu : l'agent extrait la version Rust **1.98.0** et l'édition Cargo **2024** du livre officiel, et ces valeurs correspondent au texte récupéré. Une page d'installation hors Wikipédia a également été lue via WebDriver ; la lecture d'un passage d'article à l'offset 8 000 renvoie les 4 000 octets suivants. Les essais ont rencontré un GPU limité à 300 MHz sur batterie ; le délai modèle suit maintenant le budget global. La disponibilité d'un index généraliste par API reste à vérifier avec une clé Brave.

## Vitesse mesurée

Sur Qwen3-4B-Instruct-2507, Candle 0.11.0, float16, une requête à la fois, après échauffement :

| Alimentation | Débit API moyen | Durée pour 128 tokens | Premier contenu SSE |
|---|---:|---:|---:|
| Batterie | 6,76 tokens/s | 18,94 s | 184 ms |
| Secteur | 40,60 tokens/s | 3,15 s | 42 ms |

Chaque débit est mesuré sur trois réponses de 128 tokens, avec le même prompt et une température nulle. Les sorties sont identiques entre les deux mesures. Les durées incluent le préremplissage, le décodage et les échanges HTTP locaux, hors chargement du modèle. Le gain observé de **6,01×** correspond au passage sur secteur, sans changement des poids ni du moteur.

Pendant la mesure sur secteur : environ 109 W, 8 505 Mio de VRAM utilisés sur 16 384 Mio, activité GPU moyenne 92 %, température maximale 64 °C. Sur batterie, le système indiquait un plafond de 50 W et une fréquence SM d'environ 300 MHz. Gardez le chargeur principal branché pour retrouver les conditions de la mesure rapide. Ces chiffres concernent ce workload court ; ils ne constituent pas une garantie pour tous les prompts et contextes.

Les relevés complets sont conservés dans [la mesure sur batterie](benchmarks/qwen3-4b-candle-fp16-baseline.json) et [la mesure sur secteur](benchmarks/qwen3-4b-candle-fp16-ac.json).

## Variante locale avec vLLM

Prérequis : Linux, pilote NVIDIA fonctionnel (`nvidia-smi`), Rust et Python 3.12 pour le moteur vLLM. Le modèle doit déjà être téléchargé. Docker et Kubernetes ne sont pas nécessaires pour ce mode.

Le moteur Python est isolé hors du dépôt, dans un répertoire temporaire. Aucun `.venv` n'est créé dans le projet et les paquets Python système restent inchangés. Les dépendances CUDA représentent plusieurs Go ; `/tmp` peut être nettoyé au redémarrage.

```bash
python3 -m venv /tmp/custom-ai-runtime
/tmp/custom-ai-runtime/bin/pip install -r deploy/vllm-requirements.txt

cargo run -p cli -- serve --vllm-bin /tmp/custom-ai-runtime/bin/vllm
```

Par défaut, le modèle est lu depuis `/home/rustspeak/neurax-ai-lab/models/Qwen3-4B-Instruct-2507`. Pour un autre emplacement :

```bash
cargo run -p cli -- serve \
  --vllm-bin /tmp/custom-ai-runtime/bin/vllm \
  --model /chemin/vers/Qwen3-4B-Instruct-2507
```

La CLI vérifie les fichiers requis et les ports disponibles, lance vLLM puis expose la gateway sur `127.0.0.1:8080`. `/readyz` répond `503` tant que le modèle charge et `200` lorsque vLLM est disponible. Un arrêt du moteur fait quitter la CLI avec une erreur ; Ctrl+C ou SIGTERM arrête le lancement local.

## Interroger l’API HTTP

Les exemples suivants fonctionnent avec le moteur Candle comme avec vLLM. Dans un second terminal :

```bash
curl --fail http://127.0.0.1:8080/readyz
curl --fail http://127.0.0.1:8080/v1/models

curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b-instruct","messages":[{"role":"user","content":"Bonjour ! Présente-toi en une phrase."}],"max_tokens":128,"temperature":0}'
```

Pour recevoir la réponse progressivement :

```bash
curl --fail -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b-instruct","messages":[{"role":"user","content":"Compte de un à cinq en français."}],"max_tokens":128,"stream":true}'
```

Les options sont détaillées dans `cargo run -p cli -- serve --help` : adresse publique (`--bind`), port vLLM (`--upstream-port`), nom du modèle (`--served-model-name`) et contexte (`--max-model-len`).

## Profil GPU vLLM

| Élément | Configuration locale |
|---|---|
| GPU | NVIDIA Quadro RTX 5000, 16 Go VRAM, Turing sm_75 |
| Moteur | vLLM 0.10.2, V0, XFormers |
| PyTorch | 2.8.0, bibliothèques CUDA 12.8 |
| Poids | Qwen3-4B-Instruct-2507, conversion de bfloat16 en float16 au chargement |
| Contexte au démarrage | 8 192 tokens |
| Concurrence maximale | 4 séquences |
| Mémoire GPU cible | 90 % |

Ce profil utilise `--enforce-eager` pour démarrer sans capture de graphes CUDA. Les performances et les contextes plus longs restent à mesurer. Il charge exclusivement les fichiers locaux (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).

Le choix float16/XFormers pour Turing suit le [code de sélection CUDA de vLLM 0.10.2](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/platforms/cuda.py). Les dépendances sont fixées dans [deploy/vllm-requirements.txt](deploy/vllm-requirements.txt).

## Gateway autonome

Le binaire `inference` inclut déjà la gateway. Pour utiliser un autre moteur déjà lancé :

```bash
VLLM_UPSTREAM_URL=http://127.0.0.1:8000 \
GATEWAY_BIND_ADDR=127.0.0.1:8080 \
cargo run -p gateway
```

| Variable | Valeur par défaut | Usage |
|---|---|---|
| `VLLM_UPSTREAM_URL` | `http://127.0.0.1:8000` | Adresse du moteur pour la gateway autonome |
| `GATEWAY_BIND_ADDR` | `0.0.0.0:8080` | Écoute de la gateway autonome |
| `GATEWAY_API_KEY` | absente | Si définie, impose `Authorization: Bearer …` sur `/v1/*` |

`cli serve` utilise ses options `--bind` et `--upstream-port` et reprend `GATEWAY_API_KEY`. Une clé vide est refusée. La gateway ne transmet pas cette clé à vLLM.

- `/healthz` vérifie que la gateway répond, indépendamment du moteur.
- `/readyz` vérifie la santé de vLLM, avec un délai maximal de trois secondes.
- `/v1/*` transmet la méthode, le corps, les paramètres d'URL et les réponses, y compris les flux SSE et les erreurs du moteur. Les corps de requêtes restent limités à 2 Mio par Axum.

## Kubernetes

Les manifests dans `deploy/k8s` décrivent le namespace `custom-ai`, vLLM sur un GPU et la gateway exposée en NodePort `30080`. Ce chemin nécessite encore un cluster k3s, l'intégration NVIDIA, une image de la gateway et un déploiement vérifié. Le chemin du modèle dans le volume `hostPath` doit exister sur le nœud.

Commandes déjà disponibles, après préparation du cluster et de l'image :

```bash
cargo run -p cli -- deploy
cargo run -p cli -- status
cargo run -p cli -- logs vllm --follow
cargo run -p cli -- scale gateway 1
```

Les manifests utilisent le namespace fixe `custom-ai`. L'option `--namespace` de la CLI sélectionne celui des commandes `status`, `logs` et `scale` ; elle ne réécrit pas les manifests lors de `deploy`.

## Vérification du code

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

Les tests de la gateway ouvrent des sockets de boucle locale. La feature CUDA est optionnelle : les tests du protocole et de la gateway fonctionnent sans GPU. Les tests vérifient notamment l'authentification, la fidélité des requêtes, les erreurs vLLM, la disponibilité et l'arrivée du premier événement SSE avant la fin du flux.

## Avancement

- [x] Workspace Rust : gateway Axum, CLI Clap et moteur Candle optionnel.
- [x] Commande locale `serve` et profil GPU Turing.
- [x] Proxy HTTP avec streaming SSE, authentification optionnelle et readiness.
- [x] Tests de régression : gateway, lancement local et protocole JSON/SSE.
- [x] Présence et taille complète des trois fichiers safetensors vérifiées.
- [x] Première génération réelle en français sur le GPU avec Candle.
- [x] Génération réelle et streaming via la gateway intégrée.
- [ ] Installation complète et validation GPU du chemin vLLM (droits Docker et téléchargements à terminer).
- [ ] Construction de l'image de la gateway.
- [ ] Cluster k3s avec GPU NVIDIA et déploiement vérifié.
- [ ] Métriques et limitation de débit dans la gateway.

Validation locale du 23 novembre 2026 : réponse JSON « Bonjour. », flux SSE
« Un, deux, trois, quatre, cinq. », requête répétée identique après réinitialisation
du cache KV, préremplissage avec un prompt de 216 tokens, dépassement de contexte
refusé en HTTP 400 et moteur toujours disponible ensuite. Les vingt-neuf tests Rust,
le formatage et Clippy (dont la feature CUDA) passent. Le chemin vLLM n'a pas
encore été validé sur cette machine.

## Robustesse et contexte étendu

Les tests de panne contrôlés couvrent les erreurs du générateur, les paniques, un consommateur lent, l'annulation d'un travail en attente, la saturation de la file et l'indisponibilité pendant la récupération. Une erreur récupérable réinitialise les caches ; une perte de contexte CUDA reste fatale et ne doit pas être annoncée comme prête. Les erreurs JSON sont renvoyées sous forme structurée.

Le rappel des codes PARIS7, LYON8 et LILLE9 placés au début, au milieu et à la fin d'un document de **28 535 tokens** a été vérifié sur GPU, avec 15 801 Mio de VRAM au pic dans la configuration de test. Les essais à 32 768, 40 960, 45 056 et 49 152 tokens ont provoqué des refus d'allocation : ils ne sont pas les réglages par défaut retenus pour cette carte. Les résultats sont conservés dans [les mesures de limite mémoire](benchmarks/qwen3-context-limits.json).

La validation finale par `cli chat --file`, avec le contexte par défaut de 28 672 tokens et le budget de sortie automatique, retrouve aussi les trois codes : 130,05 s pour ce document de 28 535 tokens, avec un pic de 15 193 Mio de VRAM. Voir [le résultat du test par fichier](benchmarks/qwen3-final-file.json). Le budget est partagé entre l'entrée et la sortie ; le modèle peut terminer sa réponse avant de l'épuiser.

Le débit sur le prompt court de référence est maintenu à **40,39 tokens/s**, contre 40,08 avant modification ; cet écart faible ne constitue pas une accélération significative du décodage. Le texte généré est identique. Voir [avant](benchmarks/qwen3-before-improvements.json) et [après](benchmarks/qwen3-after-improvements.json).

Sur le même document de 3 616 tokens, avec la même réponse de deux tokens, les trois requêtes chaudes passent en moyenne de **3,03 s à 1,85 s** (**1,64×**). L'API confirme 2 048 tokens de préfixe réutilisés et le texte de réponse est identique. La première requête froide reste proche de la référence : 3,10 s contre 3,06 s. Ce gain concerne la réutilisation d'un historique commun, pas tous les prompts. Voir [avant](benchmarks/qwen3-prefix-before.json) et [après](benchmarks/qwen3-prefix-after.json). Le snapshot KV de 2 048 tokens représente environ 288 Mio ; il n'est pas conservé lorsque le prompt dépasse la moitié du contexte. `--prefix-cache-tokens 0` le désactive pour comparaison.

Les [essais de panne sur l'API réelle](benchmarks/qwen3-controlled-faults.json) vérifient les refus de données invalides, les dépassements, la déconnexion en cours de streaming et une surcharge bornée. Après fermeture du flux, la requête suivante répond en 0,12 s ; après saturation, elle répond en 0,13 s. Les tests Rust injectent en plus des erreurs du générateur, des paniques et un consommateur lent.

Les mesures ont été effectuées avec le même moteur FP16 sur secteur, une seule génération active, sans modifier les poids du modèle. Les détails du workload, de la compilation et des relevés GPU sont conservés dans les fichiers JSON.
