# E-Mail: /scratch-Verzeichnis wird nicht angelegt (Account cbhaskara)

---

**Betreff:** SLURM-Cluster – /scratch-Verzeichnis nicht beschreibbar auf dgx_01 (Account cbhaskara)

Hallo,

bei Jobs auf der Partition `dgx_01` kann für meinen Account (`cbhaskara`) kein
lokales `/scratch`-Verzeichnis angelegt werden. Laut Dokumentation (Abschnitt 4)
sollte bei Jobstart automatisch ein Verzeichnis `/scratch/<username>_<jobid>`
erstellt und dem Nutzer gehören — das passiert bei mir nicht.

**Reproduktion mit einem sauberen sbatch-Job** (keine interaktive Session,
kein `salloc`/`srun`-Attach, also kein Sonderfall):

```
$ sbatch slurm_selftest.sh
Submitted batch job 3197

$ sacct -j 3197 --format=JobID,JobName,State,ExitCode,Elapsed
JobID           JobName      State ExitCode    Elapsed
------------ ---------- ---------- -------- ----------
3197         animate-s+     FAILED      1:0   00:00:01

$ cat animate-selftest_3197.err
mkdir: cannot create directory '/scratch/cbhaskara_3197': Permission denied
```

Der Job scheitert also innerhalb einer Sekunde direkt beim `mkdir` auf das vom
System erwartete Scratch-Verzeichnis.

**Zusätzlicher Befund, der vermutlich zusammenhängt:** In interaktiven Sessions
auf dem `dgx`-Node zeigt die Shell statt meines Benutzernamens `I have no
name!` an (z. B. `I have no name!@dgx:~$`), während `whoami` auf den
Submit-Nodes (`sl-sn01`/`sl-sn02`) korrekt `cbhaskara` liefert. Das deutet auf
ein Problem bei der Namens-/UID-Auflösung (NSS) speziell auf dem `dgx`-Node
hin — was auch erklären würde, warum das Anlegen von `/scratch/<user>_<jobid>`
fehlschlägt, wenn dabei intern der Benutzername statt der reinen UID verwendet
wird.

Zusammengefasst:
- Login/`whoami` auf sl-sn01, sl-sn02: funktioniert korrekt (`cbhaskara`)
- Shell auf `dgx` (interaktiv, z. B. via `srun --pty bash`): zeigt `I have no
  name!` statt `cbhaskara`
- `/scratch/cbhaskara_<jobid>` auf `dgx`: "Permission denied", reproduzierbar
  sowohl interaktiv als auch über `sbatch` (Job 3197)

Könnten Sie prüfen, ob mein Account auf dem `dgx`-Node korrekt provisioniert
ist (UID/GID-Mapping, Home-/Scratch-Berechtigungen)? Ohne funktionierendes
`/scratch` kann ich keine Jobs ausführen, die lokalen SSD-Speicher benötigen
(in meinem Fall der Download eines ca. 46 GB großen Modells, das laut
Dokumentation nicht über den geteilten NFS-Filer laufen soll).

Vielen Dank und viele Grüße
Chanakya Advaith Bhaskara (cbhaskara)

---

## Hinweis für dich

- Trage den Empfänger ein (SLURM-Admin/IT-Support der HAW Landshut).
- `slurm_selftest.sh` liegt unter `~/magicmirror/gesture_system/` auf dem
  Cluster — falls die Admins den Job selbst nachvollziehen wollen, ist er
  direkt einsatzbereit.
- Sobald das Problem behoben ist, kannst du direkt mit
  `sbatch slurm_selftest.sh` weitermachen — daran ändert sich nichts.
