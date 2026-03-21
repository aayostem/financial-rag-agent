{{/*
Expand the name of the chart.
*/}}
{{- define "financial-rag-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "financial-rag-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s" $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Chart label — used in selector/matchLabels so it must be stable.
*/}}
{{- define "financial-rag-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{- define "financial-rag-agent.labels" -}}
helm.sh/chart: {{ include "financial-rag-agent.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{/*
Selector labels — stable subset used in matchLabels.
Must NOT include chart version (changes on upgrade → selector mismatch).
*/}}
{{- define "financial-rag-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "financial-rag-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Full image reference helper.
Usage: {{ include "financial-rag-agent.image" (dict "global" .Values.global "image" .Values.api.image) }}
*/}}
{{- define "financial-rag-agent.image" -}}
{{- $registry := .global.image.registry -}}
{{- $repo     := .image.repository -}}
{{- $tag      := .image.tag | default "latest" -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end }}

{{/*
Pod-level annotations — merges global.podAnnotations with per-component overrides.
*/}}
{{- define "financial-rag-agent.podAnnotations" -}}
{{- $merged := merge (default dict .componentAnnotations) .Values.global.podAnnotations -}}
{{- toYaml $merged }}
{{- end }}
